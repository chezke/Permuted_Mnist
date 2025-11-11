import time
import tracemalloc
import numpy as np

def evaluate_agent(env, agent, seed=42, verbose=True):
    """
    Evaluate an agent across all tasks in the Permuted-MNIST environment.

    Returns:
        accuracies: list of accuracy per task
        times: list of training+evaluation time per task (seconds)
        peak_mem_mb: peak memory usage in MB
    """

    accuracies = []
    times = []

    env.reset()
    env.set_seed(seed)
    tracemalloc.start()

    task_id = 1
    while True:
        task = env.get_next_task()
        if task is None:
            break

        # Reset agent before new task
        agent.reset()

        start_time = time.time()
        agent.train(task["X_train"], task["y_train"])
        preds = agent.predict(task["X_test"])
        acc = env.evaluate(preds, task["y_test"])
        elapsed = time.time() - start_time

        accuracies.append(acc)
        times.append(elapsed)

        if verbose:
            print(f"Task {task_id}: Acc = {acc:.2%}, Time = {elapsed:.2f}s")

        task_id += 1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mem_mb = peak / (1024 * 1024)

    if verbose:
        print("\nSummary")
        print(f"Mean accuracy: {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}")
        print(f"Total time: {np.sum(times):.2f}s")
        print(f"Peak memory: {peak_mem_mb:.2f} MB")

    return {
        "accuracies": np.array(accuracies),
        "times": np.array(times),
        "peak_mem_mb": peak_mem_mb,
    }