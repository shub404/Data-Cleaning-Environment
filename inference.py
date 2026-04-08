import requests
import json
import os
import time
import traceback
from openai import OpenAI

# ============================================================
# URL ROUTING (CRITICAL):
#   ENV_URL  = local environment server (FastAPI on port 7860)
#   API_BASE_URL = LiteLLM proxy injected by the validator
#   API_KEY  = proxy key injected by the validator
# ============================================================
ENV_URL = "http://localhost:7860"
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

print(f"[CONFIG] ENV_URL      = {ENV_URL}")
print(f"[CONFIG] API_BASE_URL = {API_BASE_URL}")
print(f"[CONFIG] MODEL_NAME   = {MODEL_NAME}")
print(f"[CONFIG] API_KEY      = {'SET (' + API_KEY[:8] + '...)' if API_KEY else 'NOT SET'}")

# OpenAI client — uses the validator's LiteLLM proxy URL directly
# The validator says: base_url=os.environ["API_BASE_URL"]
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY if API_KEY else "no-key",
)


# -------------------- HELPERS -------------------- #

def env_request(method, path, **kwargs):
    """Make a request to the LOCAL environment server. Never to the LLM proxy."""
    url = f"{ENV_URL}{path}"
    try:
        print(f"[ENV] {method.upper()} {url}")
        res = requests.request(method, url, timeout=15, **kwargs)
        print(f"[ENV] Status: {res.status_code}")
        return res
    except Exception as e:
        print(f"[ENV ERROR] {method.upper()} {url} -> {e}")
        return None


def parse_json(res):
    """Safely parse JSON from a response."""
    if res is None:
        return None
    try:
        return res.json()
    except Exception as e:
        print(f"[JSON ERROR] {e}")
        print(f"[JSON ERROR] Raw: {res.text[:500]}")
        return None


def get_action_from_llm(observation, reflection=""):
    """
    ALWAYS call the LLM via the injected proxy.
    Falls back to heuristic ONLY if LLM call fails, not if key is missing.
    """
    system_msg = (
        "You are a data cleaning RL agent. Based on the observation, decide the next action. "
        "Available actions: remove_duplicates, fill_missing (needs column), "
        "normalize_text (needs column), done. "
        "Return ONLY a JSON object with keys: action_type, column (or null), reasoning."
    )
    user_msg = f"Observation: {json.dumps(observation)}"
    if reflection:
        user_msg += f"\nReflection: {reflection}"
    user_msg += "\nDecide the next cleaning action. Return JSON only."

    try:
        print(f"[LLM] Calling {MODEL_NAME} via {API_BASE_URL}...")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        print(f"[LLM] Response: {result}")
        return result
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        print("[LLM] Falling back to heuristic...")
        return heuristic_action(observation)


def heuristic_action(observation):
    """Rule-based fallback if LLM fails."""
    issues = observation.get("issues", [])
    if not issues:
        return {"action_type": "done", "column": None}
    issue = issues[0]
    if issue == "duplicates":
        return {"action_type": "remove_duplicates", "column": None}
    if "missing" in issue and ":" in issue:
        return {"action_type": "fill_missing", "column": issue.split(":")[1]}
    if "format" in issue and ":" in issue:
        return {"action_type": "normalize_text", "column": issue.split(":")[1]}
    return {"action_type": "done", "column": None}


# -------------------- MAIN -------------------- #

def main():
    print("[START] Agentic Data Cleaning Session")

    try:
        # ---- RESET the environment ----
        res = env_request("GET", "/reset?difficulty=hard")
        if not res:
            print("[FATAL] Cannot reach environment server at " + ENV_URL)
            return
        if res.status_code != 200:
            print(f"[FATAL] /reset returned {res.status_code}: {res.text[:500]}")
            return

        data = parse_json(res)
        if not data:
            print("[FATAL] /reset response is not valid JSON")
            return
        if "observation" not in data:
            print(f"[FATAL] /reset missing 'observation'. Keys: {list(data.keys())}")
            print(f"[FATAL] Body: {json.dumps(data)[:800]}")
            return

        obs = data["observation"]
        start_data = obs.get("data_preview", [])
        total_reward = 0
        prev_reward = 0
        reflection = ""

        # ---- Step loop ----
        for i in range(1, 11):
            print(f"\n--- Step {i} ---")

            action = get_action_from_llm(obs, reflection)

            # Send action to environment
            step_payload = {
                "action_type": action.get("action_type", "done"),
                "column": action.get("column"),
            }
            res = env_request("POST", "/step", json=step_payload)
            if not res or res.status_code != 200:
                print(f"[ERROR] /step failed at step {i}")
                break

            res_data = parse_json(res)
            if not res_data:
                print(f"[ERROR] /step response not valid JSON at step {i}")
                break

            reward = res_data.get("reward", 0)
            done = res_data.get("done", False)
            info = res_data.get("info", {})
            reasoning = info.get("reasoning", "n/a")

            print(f"[STEP {i}] action={step_payload['action_type']} col={step_payload.get('column')} reward={reward} done={done}")
            print(f"[STEP {i}] reasoning: {reasoning}")

            # Reflection for next step
            if reward < prev_reward:
                reflection = "Previous action degraded quality. Try a different approach."
            else:
                reflection = ""
            prev_reward = reward

            if "observation" not in res_data:
                print("[WARN] No observation in step response, ending early")
                total_reward = reward
                break

            obs = res_data["observation"]

            if done:
                total_reward = reward
                break

            time.sleep(0.3)

        print(f"\n[END] Final score: {total_reward:.4f}")

        # ---- HTML report ----
        try:
            end_preview = obs.get("data_preview", []) if isinstance(obs, dict) else []
            html = f"""<!DOCTYPE html>
<html><head><title>Scorecard</title></head><body>
<h1>DataClean-RL Report</h1>
<h2>Final Score: {total_reward:.4f}</h2>
<h3>Before</h3><pre>{json.dumps(start_data[:5], indent=2)}</pre>
<h3>After</h3><pre>{json.dumps(end_preview[:5], indent=2)}</pre>
</body></html>"""
            with open("integrity_report.html", "w") as f:
                f.write(html)
        except Exception as e:
            print(f"[HTML ERROR] {e}")

    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
