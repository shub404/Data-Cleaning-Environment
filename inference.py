import requests
import json
import os
import time
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

console = Console()

# Official OpenEnv Env Vars
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# OpenAI client configured with hackathon variables
# We assume the base URL for the model is either API_BASE_URL or a derived path
client = OpenAI(
    base_url=f"{API_BASE_URL}/v1" if API_BASE_URL.endswith("/") is False else f"{API_BASE_URL}v1",
    api_key=HF_TOKEN if HF_TOKEN else "no-token"
)

def log_start():
    # OFFICIAL REQUIRED LOG
    print("[START]")
    console.print(Panel("[bold cyan]🚀 Agentic Cleaning Session Started[/bold cyan]", expand=False))

def log_step(step, action, reward, reasoning, done):
    # OFFICIAL REQUIRED LOG (Strict Format)
    print(f"[STEP] step={step} action={action.get('action_type')} col={action.get('column')} reward={reward} done={done} reasoning='{reasoning}'")
    
    # Rich UI for Human Viewers
    col_str = f"({action.get('column')})" if action.get('column') else ""
    console.print(f"  [bold yellow]Step {step}[/bold yellow] | Action: [green]{action.get('action_type')}{col_str}[/green]")
    console.print(f"  [dim]Reasoning: {reasoning}[/dim]")
    console.print(f"  [bold]Current Score:[/bold] {reward}")
    console.print("-" * 50)

def log_end(final_score):
    # OFFICIAL REQUIRED LOG
    print(f"[END] score={final_score:.4f}")
    if final_score > 0.9:
        console.print(Panel(f"[bold green]✅ Final DQS Score: {final_score:.4f}[/bold green]", expand=False))
    else:
        console.print(Panel(f"[bold red]⚠️ Final DQS Score: {final_score:.4f}[/bold red]", expand=False))

def generate_html_scorecard(start_data, end_data, final_score):
    """Creates a side-by-side comparison report."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Scorecard</title><style>
    body{{font-family:sans-serif;margin:40px;background:#f8f9fa;}}
    .score{{font-size:32px;font-weight:bold;margin:20px;padding:20px;border-radius:12px;text-align:center;}}
    .green{{background:#d4edda;color:#155724;}} .red{{background:#f8d7da;color:#721c24;}}
    .box{{background:white;padding:20px;border-radius:12px;box-shadow:0 4px 6px rgba(0,0,0,0.1);overflow:auto;width:45%;}}
    </style></head><body>
    <h1>DataClean-RL Performance Report</h1>
    <div class="score {'green' if final_score > 0.8 else 'red'}">Final Score: {final_score:.4f}</div>
    <div style="display:flex;justify-content:space-between;">
    <div class="box"><h3>Before</h3><pre>{json.dumps(start_data[:5], indent=2)}</pre></div>
    <div class="box"><h3>After</h3><pre>{json.dumps(end_data[:5], indent=2)}</pre></div>
    </div></body></html>"""
    with open("integrity_report.html", "w") as f: f.write(html)
    console.print("[bold blue]📝 Integrity report generated -> integrity_report.html[/bold blue]")

def get_action_from_llm(observation, reflection_prompt=""):
    """
    Standard OpenAI call using Hackathon environment variables.
    """
    if not HF_TOKEN:
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
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"action_type": "done"}

def main():
    log_start()
    
    # Init Env
    res = requests.get(f"{API_BASE_URL}/reset?difficulty=hard")
    data = res.json()
    obs = data["observation"]
    start_data = obs["data_preview"]
    
    total_reward = 0
    prev_reward = 0
    reflection = ""
    
    with console.status("[bold green]Cleaning Data...") as status:
        for i in range(1, 11):
            action = get_action_from_llm(obs, reflection)
            
            res = requests.post(f"{API_BASE_URL}/step", json=action)
            res_data = res.json()
            
            reward = res_data["reward"]
            done = res_data["done"]
            reason = res_data["info"].get("reasoning", "n/a")
            
            log_step(i, action, reward, reason, done)
            
            # Reflection Logic
            if reward < prev_reward:
                reflection = "Previous action degraded quality. Try a different approach."
                console.print(f"  [bold red]🧠 Reflection Loop:[/bold red] Agent detected quality drop.")
            else:
                reflection = ""
                
            prev_reward = reward
            obs = res_data["observation"]
            
            if done:
                total_reward = reward
                break
            time.sleep(0.3)
            
    log_end(total_reward)
    generate_html_scorecard(start_data, obs["data_preview"], total_reward)

if __name__ == "__main__":
    main()