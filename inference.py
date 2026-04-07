import requests
import time

BASE_URL = "http://localhost:8000"

MAX_STEPS = 5


def log_start():
    print("[START]")


def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward} done={done}")

def log_end(score):
    print(f"[END] score={score:.2f}")


def main():
    log_start()

    res = requests.get(f"{BASE_URL}/reset")
    data = res.json()

    total_reward = 0

    for step in range(1, MAX_STEPS + 1):

        if "duplicates" in data["observation"]["issues"]:
            action = {"action_type": "remove_duplicates"}

        elif "missing_values" in data["observation"]["issues"]:
            action = {"action_type": "fill_missing"}

        elif "formatting" in data["observation"]["issues"]:
            action = {"action_type": "normalize_text"}

        else:
            action = {"action_type": "done"}

        res = requests.post(f"{BASE_URL}/step", json=action)
        data = res.json()

        reward = data["reward"]
        done = data["done"]

        total_reward += reward

        log_step(step, action, reward, done)

        if done:
            break

        time.sleep(0.5)

    score = max(0, min(1, total_reward))
    log_end(score)


if __name__ == "__main__":
    main()