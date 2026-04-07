from fastapi import FastAPI
from env.environment import DataCleaningEnv
from env.models import Action

import gradio as gr

def my_function(x):
    return x

iface = gr.Interface(fn=my_function, inputs="text", outputs="text")

iface.launch()

app = FastAPI()

env = DataCleaningEnv()


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