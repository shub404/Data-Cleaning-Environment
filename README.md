---
title: DataClean RL
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🚀 DataClean-RL: Agentic Cleaning Environment v2.0
**DataClean-RL** is a professional-grade Reinforcement Learning environment for automated data cleaning. It uses a **Strategic AI Planner** to analyze messy datasets and systematically fix missing values, formatting inconsistencies, duplicates, and outliers.

## ✨ Features (Hackathon Optimized)

-   **Multi-Scenario Data Factory**: Generates realistic "dirty" datasets for **Medical**, **Finance**, and **E-commerce** domains using dynamic difficulty (Easy/Medium/Hard).
-   **Strategic AI Planner**: An agentic loop that reasons about the data structure and plans a multi-step cleaning course.
-   **Data Quality Score (DQS)**: A multi-objective reward function considering:
    -   `Completeness (40%)`: Missing values penalty.
    -   `Consistency (30%)`: Formatting and casing normalization.
    -   `Uniqueness (20%)`: Duplicate records penalty.
    -   `Integrity (10%)`: Data loss (excessive deletion) penalty.
-   **Observability**: Detailed reason logs for every action taken by the agent.
-   **Docker Ready**: Optimized for deployment on Hugging Face Spaces.

---

## 🛠 Usage

### 🚀 1. Local Deployment (Docker)

```bash
docker build -t dataclean-rl .
docker run -p 7860:7860 dataclean-rl
```

### 🧠 2. Running the AI Planner

Set your API Key:
```bash
set OPENAI_API_KEY=your_key
```

Run the Strategic Planner:
```bash
python inference_planner.py
```

### 📊 3. Benchmarking

Run a 10-trial benchmark on "Hard" tasks:
```bash
python benchmark.py
```

---

## 🏗 API Specification (Hugging Face Space)

You can explore the environment live at the following endpoints:

-   **Interactive Dashboard**: [shubpaste404-data-cleaning-env.hf.space/docs](https://shubpaste404-data-cleaning-env.hf.space/docs) (Swagger UI)
-   **Reset / Init Task**: [shubpaste404-data-cleaning-env.hf.space/reset](https://shubpaste404-data-cleaning-env.hf.space/reset)
-   **Inspect State**: [shubpaste404-data-cleaning-env.hf.space/state](https://shubpaste404-data-cleaning-env.hf.space/state)
-   **Step API**: POST request to `/step` with `Action` payload.

---

## 🏆 Winners Choice
Developed for top-tier hackathons. Optimized for clarity, performance and impact.