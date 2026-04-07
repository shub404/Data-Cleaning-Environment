import requests
import time
import pandas as pd
import numpy as np

# Configuration
BASE_URL = "http://localhost:8000"

def run_trial(difficulty="medium"):
    """Runs a single cleaning trial and returns metrics."""
    res = requests.get(f"{BASE_URL}/reset?difficulty={difficulty}")
    obs = res.json()["observation"]
    
    start_time = time.time()
    steps = 0
    
    while steps < 10:
        steps += 1
        issues = obs.get("issues", [])
        
        if not issues:
            action = {"action_type": "done", "column": None}
        else:
            first = issues[0]
            if first == "duplicates":
                action = {"action_type": "remove_duplicates", "column": None}
            elif "missing" in first:
                action = {"action_type": "fill_missing", "column": first.split(":")[1]}
            else:
                action = {"action_type": "normalize_text", "column": first.split(":")[1]}
        
        res = requests.post(f"{BASE_URL}/step", json=action)
        result = res.json()
        obs = result["observation"]
        
        if result["done"]:
            return {
                "reward": result["reward"],
                "steps": steps,
                "duration": time.time() - start_time,
                "status": "Success" if result["reward"] > 0.8 else "Partial"
            }
    
    return {"reward": 0, "steps": 10, "duration": time.time() - start_time, "status": "Timed Out"}

def run_benchmark(difficulty="hard", n_trials=5):
    print(f"\n--- BENCHMARK START: Difficulty={difficulty.upper()} | Trials={n_trials} ---")
    
    results = []
    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials}...", end="\r")
        results.append(run_trial(difficulty))
    
    df = pd.DataFrame(results)
    
    summary = {
        "Avg Reward": df["reward"].mean(),
        "Max Reward": df["reward"].max(),
        "Avg Steps": df["steps"].mean(),
        "Avg Time (s)": df["duration"].mean(),
        "Success Rate": (df["status"] == "Success").mean() * 100
    }
    
    print("\n\n--- BENCHMARK SUMMARY ---")
    for k, v in summary.items():
        print(f"{k:15}: {v:.4f}" if isinstance(v, float) else f"{k:15}: {v}")
    
    df.to_csv(f"benchmark_{difficulty}.csv", index=False)
    print(f"\nResults saved to benchmark_{difficulty}.csv")
    print("--------------------------\n")

if __name__ == "__main__":
    try:
        run_benchmark(difficulty="hard", n_trials=5)
    except Exception as e:
        print(f"Error connecting to server: {e}. Make sure 'python app.py' is running.")
