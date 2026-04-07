import requests
import json
import os
from openai import OpenAI

BASE_URL = "http://localhost:8000"

def get_cleaning_plan(observation):
    """
    Calls OpenAI to generate a multi-step cleaning plan.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[WARNING] No OPENAI_API_KEY found. Falling back to heuristic planner.")
        return mock_planner_logic(observation)

    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    You are a Data Planning Agent. You analyze issues and generate a sequence of actions.
    
    Observation:
    {json.dumps(observation, indent=2)}

    Available Actions:
    - remove_duplicates
    - fill_missing (specify column)
    - normalize_text (specify column)
    - done

    Return a JSON object with a "plan" key containing a list of actions:
    {{
        "reasoning": "Overview of the strategy",
        "plan": [
            {{"action_type": "string", "column": "string or null"}},
            ...
        ]
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a strategic data cleaning planner."},
                      {"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[ERROR] Planning failed: {e}")
        return mock_planner_logic(observation)

def mock_planner_logic(obs):
    """Heuristic planner fallback."""
    issues = obs.get("issues", [])
    plan = []
    
    for issue in issues:
        if issue == "duplicates":
            plan.append({"action_type": "remove_duplicates", "column": None})
        elif "missing_values" in issue:
            col = issue.split(":")[1]
            plan.append({"action_type": "fill_missing", "column": col})
        elif "format_inconsistency" in issue:
            col = issue.split(":")[1]
            plan.append({"action_type": "normalize_text", "column": col})
    
    plan.append({"action_type": "done", "column": None})
    return {"plan": plan, "reasoning": "Rule-based sequential planner."}

def run_planner_session(difficulty="hard"):
    print(f"\n=== Starting Multi-Step Planning Session ({difficulty.upper()}) ===")
    
    res = requests.get(f"{BASE_URL}/reset?difficulty={difficulty}")
    data = res.json()
    obs = data["observation"]
    
    full_plan = get_cleaning_plan(obs)
    plan = full_plan.get("plan", [])
    print(f"[PLANNER] Strategy: {full_plan.get('reasoning')}")
    print(f"[PLANNER] Sequence: {[step['action_type'] for step in plan]}")
    
    final_reward = 0
    for idx, step in enumerate(plan):
        print(f"Step {idx+1}: Executing {step['action_type']} on {step.get('column')}...")
        res = requests.post(f"{BASE_URL}/step", json={
            "action_type": step["action_type"],
            "column": step.get("column")
        })
        result = res.json()
        
        if result["done"]:
            final_reward = result["reward"]
            print(f"\n[DONE] Final Data Quality Score: {final_reward}")
            break
            
    print(f"========================================\n")

if __name__ == "__main__":
    run_planner_session(difficulty="hard")
