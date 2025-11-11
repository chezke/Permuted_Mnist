# hpo_my_agent1.py
import time
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.exceptions import TrialPruned

# === 按你的项目结构修改这里的导入路径 ===
# 你的 agent 提交文件保持不变；这里只“使用”它
from permuted_mnist.agent.my_agent1.agent import Agent as MyAgent1
from permuted_mnist.env import PermutedMNISTEnv  # 若路径不同，请改成你的 env 导入

TIME_LIMIT_PER_TASK = 60.0  # 秒（训练 + 预测）
RANDOM_SEED = 42

def set_global_seeds(seed=RANDOM_SEED):
    np.random.seed(seed)

def make_agent_from_trial(trial: optuna.Trial) -> MyAgent1:
    """
    用 trial 采样的超参实例化你的 Agent（不修改 agent 文件本身）。
    搜索空间偏向“≤60s/任务”的可行配置：大 batch、1~2 epochs、较强重放。
    """
    # 网络结构
    hidden0 = trial.suggest_categorical("hidden0", [512, 768, 1024, 1280])
    hidden1 = trial.suggest_categorical("hidden1", [256, 384, 512, 640])
    hidden = (hidden0, hidden1)

    # 批大小（尽量大以减少步数）
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])

    # 轮数（2 足够强，1 更稳卡时）
    epochs = trial.suggest_categorical("epochs", [1, 2])

    # 重放相关
    replay_max_samples = trial.suggest_categorical("replay_max_samples", [8000, 10000, 12000])
    replay_keep_tasks  = trial.suggest_categorical("replay_keep_tasks",  [4, 5, 6])

    # 初始重放比例（每任务会动态调整）
    base_replay_ratio  = trial.suggest_float("base_replay_ratio", 0.20, 0.40)
    slope_replay_ratio = trial.suggest_float("slope_replay_ratio", 0.08, 0.18)  # 每个任务增加量
    max_replay_ratio   = trial.suggest_float("max_replay_ratio", 0.70, 0.90)

    # 学习率初值（若 agent 支持 learning_rate_init 构造参数）
    lr_init = trial.suggest_float("learning_rate_init", 0.0015, 0.004, log=True)

    # L2 正则（若能从外部覆盖）
    alpha = trial.suggest_float("alpha", 1e-4, 5e-4, log=True)

    # —— 构造你的 Agent（不更改文件，只传参）——
    agent = MyAgent1(
        output_dim=10,
        seed=RANDOM_SEED,
        hidden=hidden,
        batch_size=batch_size,
        epochs=epochs,
        replay_max_samples=replay_max_samples,
        replay_keep_tasks=replay_keep_tasks,
        replay_ratio=base_replay_ratio,      # 每任务会被动态覆盖
        learning_rate_init=lr_init,          # 若你的 Agent 不接收该参数，请删除此行
    )

    # 尝试从外部调整 L2（如果可行）
    try:
        if hasattr(agent, "model") and hasattr(agent.model, "alpha"):
            agent.model.alpha = alpha
    except Exception:
        pass

    # 把动态重放策略临时挂在 agent 上；评测时读取
    agent._hpo_dynamic_replay = (base_replay_ratio, slope_replay_ratio, max_replay_ratio)
    return agent

def evaluate_agent(agent: MyAgent1, hard_time_limit_s: float = TIME_LIMIT_PER_TASK) -> float:
    """
    在完整任务序列上评估平均准确率。
    硬时间约束：若任一任务耗时 > hard_time_limit_s，直接剪枝该 trial。
    """
    env = PermutedMNISTEnv(seed=RANDOM_SEED)
    env.reset()
    env.set_seed(RANDOM_SEED)

    accuracies = []
    task_num = 1

    while True:
        task = env.get_next_task()
        if task is None:
            break

        X_tr, y_tr = task["X_train"], task["y_train"]
        X_te, y_te = task["X_test"],  task["y_test"]

        # —— 动态重放比例（不改 agent 文件，只改属性）——
        if hasattr(agent, "_hpo_dynamic_replay"):
            base, slope, rmax = agent._hpo_dynamic_replay
            agent.replay_ratio = min(base + slope * (task_num - 1), rmax)

        # ———— 计时开始（训练+预测）————
        start = time.perf_counter()

        # 如果你的 Agent.train 支持 time_budget_s 参数，可以传入以帮助卡时；
        # 如果不支持，就直接调用 train（仍由外层硬约束把超时 trial 剪枝）
        try:
            agent.train(X_tr, y_tr, time_budget_s=hard_time_limit_s)
        except TypeError:
            agent.train(X_tr, y_tr)

        preds = agent.predict(X_te)

        elapsed = time.perf_counter() - start

        # 硬约束：超时直接剪枝
        if elapsed > hard_time_limit_s:
            raise TrialPruned(f"Task {task_num} exceeded {hard_time_limit_s:.1f}s (elapsed={elapsed:.2f}s).")

        acc = env.evaluate(preds, y_te)
        accuracies.append(acc)
        task_num += 1

    return float(np.mean(accuracies))

def objective(trial: optuna.Trial) -> float:
    set_global_seeds(RANDOM_SEED)
    agent = make_agent_from_trial(trial)
    mean_acc = evaluate_agent(agent, hard_time_limit_s=TIME_LIMIT_PER_TASK)
    # 记录一些关键信息便于对照
    try:
        trial.set_user_attr("final_hidden", agent.model.hidden_layer_sizes)
        trial.set_user_attr("batch_size", agent.batch_size)
        trial.set_user_attr("epochs", agent.epochs)
        trial.set_user_attr("replay_ratio_eval", getattr(agent, "replay_ratio", None))
    except Exception:
        pass
    return mean_acc  # maximize

def main():
    sampler = TPESampler(seed=RANDOM_SEED, multivariate=True, constant_liar=True)
    study = optuna.create_study(
        study_name="permuted_mnist_hybrid_agent_hpo",
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
    )
    # 视机器性能调整 n_trials
    study.optimize(objective, n_trials=50, timeout=None, gc_after_trial=True, show_progress_bar=True)

    best = study.best_trial
    print("\n[OPTUNA] best value (mean acc):", best.value)
    print("[OPTUNA] best params:", best.params)
    print("[OPTUNA] user attrs:", best.user_attrs)

if __name__ == "__main__":
    main()