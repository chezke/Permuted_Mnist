import time
import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    raise ImportError("This agent requires PyTorch. Please install torch.") from e


# -----------------------------
# Internal MLP blocks (fast)
# -----------------------------
class _Block(nn.Module):
    def __init__(self, in_f, out_f, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, out_f),
            nn.BatchNorm1d(out_f),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )
    def forward(self, x):
        return self.net(x)

class _MLP(nn.Module):
    def __init__(self, in_f=784, n_classes=10, width=768, depth=2, p_drop=0.1):
        super().__init__()
        layers = [_Block(in_f, width, p_drop)]
        for _ in range(depth-1):
            layers.append(_Block(width, width, p_drop))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, n_classes)
        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.head(self.backbone(x))


class Agent:
    """Torch MLP agent (符合 reset/train/predict 规范接口)

    目标：permuted MNIST 单任务 >99% 准确率，同时整套评测 < 60s（默认设定）。
    关键：
      - 特征标准化（/255 后按特征 z-score）
      - 宽而浅的 MLP（768x2, ReLU+BN+Dropout）
      - AdamW+CosineLR，早停 + 每任务时间上限
      - 大 batch（默认 2048）加速
    """
    def __init__(
        self,
        output_dim: int = 10,
        seed: int = None,
        # 输入维度：通常是 784（28*28 扁平）
        input_dim: int = 784,
        # 速度-精度权衡参数（可按机器性能调）
        width: int = 768,
        depth: int = 2,
        dropout: float = 0.1,
        batch_size: int = 2048,
        epochs: int = 12,
        lr: float = 3e-3,
        weight_decay: float = 5e-4,
        label_smoothing: float = 0.05,
        early_stop_patience: int = 2,
        max_seconds_per_task: float = 5.0,  # 每个任务最多训练 5 秒
    ):
        self.output_dim = output_dim
        self.rng = np.random.RandomState(seed)
        self.seed = seed

        # 保存超参
        self.input_dim = input_dim
        self.width = width
        self.depth = depth
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.early_stop_patience = early_stop_patience
        self.max_seconds_per_task = max_seconds_per_task

        # 设备
        self.device = torch.device('cpu')

        # 运行时状态
        self.model = None
        self.scaler_mean_ = None
        self.scaler_std_ = None

    # ---------- 规范接口 ----------
    def reset(self):
        """Reset the agent for a new task/simulation"""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
        self.model = _MLP(
            in_f=self.input_dim,
            n_classes=self.output_dim,
            width=self.width,
            depth=self.depth,
            p_drop=self.dropout,
        ).to(self.device)
        self.scaler_mean_ = None
        self.scaler_std_ = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the agent on the provided data"""
        if self.model is None:
            self.reset()

        # --- 预处理：扁平 & 标准化 ---
        X = X_train
        if X.ndim == 3:
            # (N, 28, 28) -> (N, 784)
            X = X.reshape(X.shape[0], -1)
        X = X.astype(np.float32) / 255.0
        y = y_train.astype(np.int64).reshape(-1)

        # 按特征 z-score
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-6] = 1.0
        self.scaler_mean_ = mean
        self.scaler_std_ = std
        X = (X - mean) / std

        # to tensor
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        # 优化器/损失/调度
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = self.epochs * max(1, math.ceil(len(ds) / self.batch_size))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # 早停 + 时间上限
        start_t = time.time()
        best_loss = float('inf')
        best_state = None
        patience = 0
        step = 0

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                # 第一 epoch 的前若干步做简单 warmup
                if step < len(loader):
                    for pg in opt.param_groups:
                        pg['lr'] = self.lr * 0.4

                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                scheduler.step()

                epoch_loss += loss.item() * xb.size(0)
                step += 1

                # 时间限制：严格控制每任务开销
                if (time.time() - start_t) > self.max_seconds_per_task:
                    break

            avg_loss = epoch_loss / max(1, len(ds))

            # early stop 依据训练损失（无验证集以节省时间）
            if avg_loss + 1e-6 < best_loss:
                best_loss = avg_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            # 超时或耐心耗尽则停止
            if (time.time() - start_t) > self.max_seconds_per_task or patience >= self.early_stop_patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions on test data"""
        if self.model is None or self.scaler_mean_ is None:
            raise RuntimeError("Agent not trained or scaler not fitted.")

        X = X_test
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        X = X.astype(np.float32) / 255.0
        X = (X - self.scaler_mean_) / self.scaler_std_

        X_t = torch.from_numpy(X)
        ds = TensorDataset(X_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                logits = self.model(xb)
                preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)


# 导出名保证兼容
__all__ = ["Agent"]