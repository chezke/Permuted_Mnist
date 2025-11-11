# Permuted MNIST — MLP Agent & Evaluation Framework

This repository contains **my MLP-based agent** for the Permuted-MNIST continual learning benchmark, along with a **lightweight evaluation framework**.  
This project focuses on **agent design and reproducible experiments**, not on providing the environment itself.

---

## Installation

> The Permuted-MNIST **environment** (dataset and task generator) is not included here.  
> It must be installed separately from the upstream project.

    # 1) Clone and install this project
    git clone https://github.com/chezke/Permuted_Mnist.git
    cd Permuted_Mnist
    pip install -e .

    # 2) Install the Permuted-MNIST environment
    pip install git+https://github.com/ml-arena/permuted_mnist.git

    # 3) Install PyTorch (choose CPU/CUDA version as needed)
    pip install torch
    # For CUDA: https://pytorch.org/get-started/

**Python:** 3.9+ recommended.

---

## Repository Structure

    Permuted_Mnist/
    ├── pyproject.toml
    ├── README.md
    └── src/
        └── permuted_mnist/
            ├── __init__.py
            ├── agents/
            │   ├── __init__.py
            │   ├── Experiments/               # Experimental agents (work-in-progress)
            │   │   ├── hpo_mlp.py             # hyperparameter search for MLP agent
            │   │   ├── agent.py
            │   │   ├── agent1.py
            │   │   ├── agent3.py
            │   │   ├── agent4.py
            │   │   ├── agent5.py
            │   │   ├── agent6.py
            │   │   ├── agent_standarized.py
            │   │   ├── agent_Faulkner.py
            │   │   └── mlp.py
            │   ├── linear/
            │   │   └── agent.py               # Baseline linear classifier
            │   ├── mlp_agent/
            │   │   └── agent.py               # Main MLP continual-learning agent (The Final Agent)
            │   ├── random/
            │   │   └── agent.py               # Random baseline agent
            │   └── torch_mlp/                 # Other torch-based variants (arch/hparam tests)
            │       ├── agent.py
            │       ├── agent11.py
            │       ├── agent22.py
            │       └── agent33.py
            ├── evaluation/
            │   ├── __init__.py
            │   ├── evaluator.py               # Multi-task evaluation loop
            │   └── metrics.py                 # Accuracy, summaries, tables, optional plotting
            └── utils/
                ├── __init__.py
                ├── io.py                      # Save / load checkpoints
                ├── reproducibility.py         # Seed control
                └── timers.py                  # Time + memory measurement
                

- Each agent lives in its own directory under `agents/`.
- To add a new agent, create a new folder containing an `agent.py` with:
  - `reset()`
  - `train(X_train, y_train)`
  - `predict(X_test)`

---

## Evaluation Example

The evaluation uses a standard continual-learning loop:
- Tasks arrive sequentially
- Agent state resets between tasks (soft reset unless modified)
- Train → Predict → Evaluate

````bash

    import time, numpy as np
    from permuted_mnist.env.permuted_mnist import PermutedMNISTEnv
    from permuted_mnist.agents.mlp_agent.agent import Agent

    env = PermutedMNISTEnv()
    env.reset()
    env.set_seed(42)

    agent = Agent(output_dim=10, seed=42)

    accuracies, times = [], []
    task_num = 1

    print("Evaluating MLP Agent")
    print("=" * 50)

    while True:
        task = env.get_next_task()
        if task is None:
            break

        agent.reset()
        start = time.time()

        agent.train(task["X_train"], task["y_train"])
        preds = agent.predict(task["X_test"])
        acc = env.evaluate(preds, task["y_test"])
        elapsed = time.time() - start

        accuracies.append(acc)
        times.append(elapsed)
        print(f"Task {task_num}: Accuracy = {acc:.2%}, Time = {elapsed:.2f}s")
        task_num += 1

    print("\nSummary")
    print(f"  Mean accuracy: {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}")
    print(f"  Total time: {np.sum(times):.2f}s")
````

---

## Saving / Loading Agents
````bash
    from permuted_mnist.utils.io import save_agent, load_agent
    save_agent(agent, "checkpoints/mlp_agent.pt")
    agent2 = load_agent("checkpoints/mlp_agent.pt")
````

---

## Why This Design

- Clean **`src/`-based installable package**
- Clear module separation: **agents / evaluation / utils**
- All agents conform to one interface → easy comparison
- Evaluation is explicit + reproducible (seeded)
- Suitable for automated checks (speed, accuracy constraints)

---

## License

MIT License.  
The Permuted-MNIST environment is © ml-arena/permuted_mnist (MIT). Please cite accordingly.
