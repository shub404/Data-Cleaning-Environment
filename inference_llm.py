import requests
import json
import os
from openai import OpenAI

# Configuration
BASE_URL = "http://localhost:8000"

def get_llm_action(observation):
    """
    Calls OpenAI to decide the next cleaning action based on the observation.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[WARNING] No OPENAI_API_KEY found. Falling back to heuristic mock.")
        return mock_llm_logic(observation)

    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    You are an expert Data Cleaning Reinforcement Learning Agent.
    Your goal is to maximize the Data Quality Score.

    Observation:
    {json.dumps(observation, indent=2)}

    Available Actions:
    1. "remove_duplicates": Use if 'duplicates' is in issues. (Global)
    2. "fill_missing": Use if 'missing_values:COLUMN' is in issues. Requires 'column' parameter.
    3. "normalize_text": Use if 'format_inconsistency:COLUMN' is in issues. Requires 'column' parameter.
    4. "done": Use when all issues are resolved or no progress can be made.

    Return ONLY a JSON object:
    {{
        "reasoning": "Brief explanation of your choice",
        "action_type": "string",
        "column": "string or null"
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a precise data cleaning robot."},
                      {"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[ERROR] LLM Call failed: {e}")
        return mock_llm_logic(observation)

def mock_llm_logic(obs):
    """Heuristic fallback for demo without API key."""
    issues = obs.get("issues", [])
    if not issues:
        return {"action_type": "done", "column": None, "reasoning": "No issues detected."}
    
    first_issue = issues[0]
    if first_issue == "duplicates":
        return {"action_type": "remove_duplicates", "column": None, "reasoning": "Found duplicates."}
    elif "missing_values" in first_issue:
        col = first_issue.split(":")[1]
        return {"action_type": "fill_missing", "column": col, "reasoning": f"Filling missing values in {col}"}
    elif "format_inconsistency" in first_issue:
        col = first_issue.split(":")[1]
        return {"action_type": "normalize_text", "column": col, "reasoning": f"Normalizing text in {col}"}
    
    return {"action_type": "done", "column": None, "reasoning": "Defaulting to done."}

def run_cleaning_session(difficulty="medium"):
    print(f"\n=== Starting Data Cleaning Session ({difficulty.upper()}) ===")
    
    # 1. Reset Environment
    res = requests.get(f"{BASE_URL}/reset?difficulty={difficulty}")
    data = res.json()
    obs = data["observation"]
    
    total_reward = 0
    
    for step in range(1, 11):
        # 2. Get AI Decision
        decision = get_llm_action(obs)
        print(f"[STEP {step}] Reasoning: {decision.get('reasoning')}")
        print(f"         Action: {decision.get('action_type')} ({decision.get('column')})")
        
        # 3. Execute Step
        res = requests.post(f"{BASE_URL}/step", json={
            "action_type": decision["action_type"],
            "column": decision.get("column")
        })
        result = res.json()
        
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        
        total_reward += reward
        
        if done:
            print(f"\n[DONE] Final Data Quality Score: {reward}")
            break
            
    print(f"========================================\n")

if __name__ == "__main__":
    # You can test with different difficulties
    run_cleaning_session(difficulty="hard")
