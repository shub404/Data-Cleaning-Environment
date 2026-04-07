from fastapi import FastAPI
from env.environment import DataCleaningEnv
from env.models import Action

app = FastAPI()

env = DataCleaningEnv()


@app.get("/")
def root():
    return {"status": "running"}


@app.get("/reset")
def reset():
    obs = env.reset()
    return {
        "observation": obs.dict(),
        "done": False
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