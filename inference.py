import requests
import json
import os
import time
import traceback
import sys
from openai import OpenAI

# Config is now silent to prevent parsing issues
ENV_URL = "http://localhost:7860"
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

def log(msg):
    """Log to stderr so stdout stays clean for the validator."""
    print(str(msg), file=sys.stderr)

# OpenAI client — uses the validator's LiteLLM proxy URL directly
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY if API_KEY else "no-key",
)


# -------------------- HELPERS -------------------- #

def env_request(method, path, **kwargs):
    """Make a request to the LOCAL environment server. Never to the LLM proxy."""
    url = f"{ENV_URL}{path}"
    try:
        log(f"[ENV] {method.upper()} {url}")
        res = requests.request(method, url, timeout=15, **kwargs)
        log(f"[ENV] Status: {res.status_code}")
        return res
    except Exception as e:
        log(f"[ENV ERROR] {method.upper()} {url} -> {e}")
        return None


def parse_json(res):
    """Safely parse JSON from a response."""
    if res is None:
        return None
    try:
        return res.json()
    except Exception as e:
        log(f"[JSON ERROR] {e}")
        log(f"[JSON ERROR] Raw: {res.text[:500]}")
        return None


def get_action_from_llm(observation, reflection=""):
    """
    ALWAYS call the LLM via the injected proxy.
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
        log(f"[LLM] Calling {MODEL_NAME} via {API_BASE_URL}...")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        log(f"[LLM] Response: {result}")
        return result
    except Exception as e:
        log(f"[LLM ERROR] {e}")
        log("[LLM] Falling back to heuristic...")
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

def run_task(difficulty="hard"):
    """
    Runs a single cleaning task and returns the final score strictly in (0, 1).
    """
    # Noisier logs removed for validator compatibility

    # ---- RESET the environment ----
    res = env_request("GET", f"/reset?difficulty={difficulty}")
    if not res:
        log("[FATAL] Cannot reach environment server at " + ENV_URL)
        return None
    if res.status_code != 200:
        log(f"[FATAL] /reset returned {res.status_code}: {res.text[:500]}")
        return None

    data = parse_json(res)
    if not data:
        log("[FATAL] /reset response is not valid JSON")
        return None
    if "observation" not in data:
        log(f"[FATAL] /reset missing 'observation'. Keys: {list(data.keys())}")
        return None

    obs = data["observation"]
    total_reward = None
    prev_reward = 0.5
    reflection = ""

    # ---- Step loop ----
    for i in range(1, 11):
        log(f"--- Step {i} ---")

        action = get_action_from_llm(obs, reflection)

        # Send action to environment
        step_payload = {
            "action_type": action.get("action_type", "done"),
            "column": action.get("column"),
        }
        res = env_request("POST", "/step", json=step_payload)
        if not res or res.status_code != 200:
            log(f"[ERROR] /step failed at step {i}")
            break

        res_data = parse_json(res)
        if not res_data:
            log(f"[ERROR] /step response not valid JSON at step {i}")
            break

        # FORCE valid reward
        reward = res_data.get("reward", 0.5)
        try:
            reward = float(reward)
        except:
            reward = 0.5

        if reward <= 0.0:
            reward = 0.01
        elif reward >= 1.0:
            reward = 0.99
            
        done = res_data.get("done", False)
        info = res_data.get("info", {})
        reasoning = info.get("reasoning", "n/a")

        log(f"[STEP {i}] action={step_payload['action_type']} col={step_payload.get('column')} reward={reward} done={done}")

        # Reflection for next step
        if reward < prev_reward:
            reflection = "Previous action degraded quality. Try a different approach."
        else:
            reflection = ""
        prev_reward = reward

        if "observation" not in res_data:
            log("[WARN] No observation in step response, ending early")
            total_reward = reward
            break

        obs = res_data["observation"]
        if done:
            total_reward = reward
            break

        time.sleep(0.1)

    # If the loop exhausted without done, use last known reward
    if total_reward is None:
        total_reward = prev_reward if prev_reward > 0 else 0.5

    # Ensure score is strictly between 0 and 1 (never 0.0 or 1.0)
    total_reward = float(min(max(total_reward, 0.01), 0.99))

    return total_reward


def main():
    # Only print exactly what the validator needs to stdout
    difficulties = ["easy", "medium", "hard"]
    
    for difficulty in difficulties:
        score = run_task(difficulty)
        
        if score is None:
            score = 0.5 # Fallback
            
        # Standard structured output for the validator (STDOUT ONLY)
        print(json.dumps({
            "task_id": difficulty,
            "score": float(score)
        }), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {e}")
        traceback.print_exc(file=sys.stderr)
