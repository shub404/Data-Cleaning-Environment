from fastapi import FastAPI, Request
from env.environment import DataCleaningEnv
from env.models import Action
from typing import Optional

app = FastAPI()
env = DataCleaningEnv()

@app.get("/")
def root():
    """Welcome page with navigation for hackathon judges."""
    return {
        "project": "DataClean-RL v2.0",
        "status": "Online",
        "endpoints": {
            "interactive_docs": "/docs",
            "reset_environment": "/reset",
            "take_action": "/step",
            "check_state": "/state"
        }
    }

@app.api_route("/reset", methods=["GET", "POST"])
async def reset(request: Request, difficulty: Optional[str] = "easy"):
    # If it's a POST, we try to check JSON for difficulty, otherwise use query param
    selected_difficulty = difficulty
    if request.method == "POST":
        try:
            body = await request.json()
            selected_difficulty = body.get("difficulty", difficulty)
        except:
            pass # Use query param or default if no JSON body
            
    obs = env.reset(difficulty=selected_difficulty)
    return {
        "observation": obs.dict(),
        "done": False,
        "difficulty_selected": selected_difficulty
    }

@app.api_route("/state", methods=["GET", "POST"])
def state():
    """Returns the current raw state of the environment."""
    return {
        "data_shape": env.data.shape if env.data is not None else None,
        "step_count": env.steps,
        "is_done": env.done
    }

@app.post("/step")
def step(action: Action):
    result = env.step(action)
    return {
        "observation": result["observation"].dict(),
        "reward": result["reward"],
        "done": result["done"],
        "info": result["info"]
    }