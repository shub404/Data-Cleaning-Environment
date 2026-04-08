import requests
import json
import os
import time
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
import traceback

console = Console()

# Env Vars - OpenEnv sets API_BASE_URL for the LLM proxy, NOT the environment server.
# The environment server runs locally inside the container (port 7860).
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("API_KEY")

# OpenAI client configured with hackathon variables
client = OpenAI(
    base_url=f"{API_BASE_URL}/v1" if not API_BASE_URL.endswith("/") else f"{API_BASE_URL}v1",
    api_key=API_KEY if API_KEY else "no-token"
)

# ---------------- SAFE HELPERS ---------------- #

def safe_request(method, url, **kwargs):
    try:
        print(f"[HTTP] {method.upper()} {url}")
        res = requests.request(method, url, timeout=10, **kwargs)
        print(f"[HTTP] Status: {res.status_code}")
        if res.status_code != 200:
            print(f"[HTTP ERROR] Non-200 response: {res.status_code}")
            print(f"[HTTP ERROR] Body: {res.text[:500]}")
        return res
    except requests.exceptions.RequestException as e:
        print(f"[NETWORK ERROR] {method.upper()} {url} -> {e}")
        return None

def safe_json(res):
    if res is None:
        print("[ERROR] Cannot parse JSON from None response")
        return None
    try:
        data = res.json()
        print(f"[JSON] Keys: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")
        return data
    except Exception as e:
        print(f"[ERROR] JSON parse failed: {e}")
        print(f"[ERROR] Raw response ({res.status_code}): {res.text[:500]}")
        return None

# ---------------- LOGGING ---------------- #

def log_start():
    print("[START]")
    console.print(Panel("[bold cyan]🚀 Agentic Cleaning Session Started[/bold cyan]", expand=False))

def log_step(step, action, reward, reasoning, done):
    print(f"[STEP] step={step} action={action.get('action_type')} col={action.get('column')} reward={reward} done={done} reasoning='{reasoning}'")
    
    col_str = f"({action.get('column')})" if action.get('column') else ""
    console.print(f"  [bold yellow]Step {step}[/bold yellow] | Action: [green]{action.get('action_type')}{col_str}[/green]")
    console.print(f"  [dim]Reasoning: {reasoning}[/dim]")
    console.print(f"  [bold]Current Score:[/bold] {reward}")
    console.print("-" * 50)

def log_end(final_score):
    print(f"[END] score={final_score:.4f}")
    if final_score > 0.9:
        console.print(Panel(f"[bold green]✅ Final DQS Score: {final_score:.4f}[/bold green]", expand=False))
    else:
        console.print(Panel(f"[bold red]⚠️ Final DQS Score: {final_score:.4f}[/bold red]", expand=False))

# ---------------- HTML REPORT ---------------- #

def generate_html_scorecard(start_data, end_data, final_score):
    try:
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Scorecard</title></head>
        <body>
        <h1>DataClean-RL Performance Report</h1>
        <h2>Final Score: {final_score:.4f}</h2>
        <h3>Before</h3>
        <pre>{json.dumps(start_data[:5], indent=2)}</pre>
        <h3>After</h3>
        <pre>{json.dumps(end_data[:5], indent=2)}</pre>
        </body>
        </html>
        """
        with open("integrity_report.html", "w") as f:
            f.write(html)
    except Exception as e:
        print("[HTML ERROR]", str(e))

# ---------------- LLM ---------------- #

def get_action_from_llm(observation, reflection_prompt=""):
    """
    Standard OpenAI call using Hackathon environment variables.
    """
    if not API_KEY:
        # Heuristic fallback if no token provided
        issues = observation.get("issues", [])
        if not issues: return {"action_type": "done"}
        issue = issues[0]
        if issue == "duplicates": return {"action_type": "remove_duplicates"}
        if "missing" in issue: return {"action_type": "fill_missing", "column": issue.split(":")[1]}
        return {"action_type": "normalize_text", "column": issue.split(":")[1]}

    prompt = f"Observation: {json.dumps(observation)}. {reflection_prompt} Decide next cleaning action."
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print("[LLM ERROR]", str(e))
        return {"action_type": "done"}

# ---------------- MAIN ---------------- #

def main():
    log_start()

    try:
        # RESET
        res = safe_request("GET", f"{ENV_URL}/reset?difficulty=hard")
        if not res:
            print("[FATAL] Reset request failed — is the server running?")
            print(f"[FATAL] Tried to reach: {ENV_URL}")
            return

        if res.status_code != 200:
            print(f"[FATAL] Reset returned HTTP {res.status_code}")
            print(f"[FATAL] Body: {res.text[:500]}")
            return

        data = safe_json(res)
        if not data:
            print("[FATAL] Could not parse reset response as JSON")
            return

        if "observation" not in data:
            print(f"[FATAL] Reset response missing 'observation' key")
            print(f"[FATAL] Available keys: {list(data.keys())}")
            print(f"[FATAL] Full response: {json.dumps(data, indent=2)[:1000]}")
            return

        obs = data["observation"]
        start_data = obs.get("data_preview", [])

        total_reward = 0
        prev_reward = 0
        reflection = ""

        with console.status("[bold green]Cleaning Data..."):
            for i in range(1, 11):

                action = get_action_from_llm(obs, reflection)

                res = safe_request("POST", f"{ENV_URL}/step", json=action)
                if not res:
                    print(f"[ERROR] Step {i} request failed")
                    break

                if res.status_code != 200:
                    print(f"[ERROR] Step {i} returned HTTP {res.status_code}")
                    print(f"[ERROR] Body: {res.text[:500]}")
                    break

                res_data = safe_json(res)
                if not res_data:
                    print(f"[ERROR] Step {i} response not valid JSON")
                    break

                reward = res_data.get("reward", 0)
                done = res_data.get("done", False)
                reason = res_data.get("info", {}).get("reasoning", "n/a")

                log_step(i, action, reward, reason, done)

                # Reflection
                if reward < prev_reward:
                    reflection = "Previous action degraded quality. Try a different approach."
                else:
                    reflection = ""

                prev_reward = reward

                if "observation" not in res_data:
                    print("[ERROR] Missing observation → terminating early")
                    total_reward = prev_reward
                    break

                obs = res_data["observation"]

                if done:
                    total_reward = reward
                    break

                time.sleep(0.3)

        log_end(total_reward)

        end_preview = obs.get("data_preview", []) if obs else []
        generate_html_scorecard(start_data, end_preview, total_reward)

    except Exception as e:
        print("[FATAL ERROR]", str(e))
        traceback.print_exc()

if __name__ == "__main__":
    main()
