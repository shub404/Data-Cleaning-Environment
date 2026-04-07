from fastapi import FastAPI
from env.environment import DataCleaningEnv
from env.models import Action

app = FastAPI()

env = DataCleaningEnv()


@app.get("/")
def root():
    return {"status": "running"}


@app.get("/reset")
def reset(difficulty: str = "easy"):
    obs = env.reset(difficulty=difficulty)
    return {
        "observation": obs.dict(),
        "done": False,
        "difficulty_selected": difficulty
    }


@app.get("/state")
def state():
    """Returns the current raw state of the environment for OpenEnv compliance."""
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