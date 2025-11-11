import time
import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torch.nn.utils import weight_norm
except Exception as e:
    raise ImportError("This agent requires PyTorch. Please install torch.") from e


# -----------------------------
# Residual Bottleneck MLP block (CPU-friendly)
# -----------------------------
class _Bottleneck(nn.Module):
    def __init__(self, in_f, out_f, hidden=None, p_drop=0.1):
        super().__init__()
        hidden = hidden or max(out_f // 2, 128)
        self.in_f = in_f
        self.out_f = out_f

        self.fc1 = weight_norm(nn.Linear(in_f, hidden, bias=True))
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act = nn.SiLU()  # smooth ReLU, works well on MNIST
        self.drop = nn.Dropout(p_drop)
        self.fc2 = weight_norm(nn.Linear(hidden, out_f, bias=True))
        self.bn2 = nn.BatchNorm1d(out_f)

        # projection for residual when dims mismatch
        self.proj = None
        if in_f != out_f:
            self.proj = weight_norm(nn.Linear(in_f, out_f, bias=False))

    def forward(self, x):
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.bn2(h)
        res = x if self.proj is None else self.proj(x)
        return self.act(h + res)


class _MLP(nn.Module):
    def __init__(self, in_f=784, n_classes=10, width=1024, depth=3, p_drop=0.1):
        super().__init__()
        layers = [_Bottleneck(in_f, width, hidden=width // 2, p_drop=p_drop)]
        for _ in range(depth - 1):
            layers.append(_Bottleneck(width, width, hidden=width // 2, p_drop=p_drop))
        self.backbone = nn.Sequential(*layers)
        self.head = weight_norm(nn.Linear(width, n_classes))

        # Kaiming init (use relu gain for SiLU too; it's close and stable)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.backbone(x))


# -----------------------------
# LR schedule: Warmup + Cosine
# -----------------------------
class WarmupCosine:
    def __init__(self, optimizer, total_steps, warmup_steps=100):
        self.optimizer = optimizer
        self.total = max(1, int(total_steps))
        self.warmup = max(0, int(warmup_steps))
        self.step_num = 0

    def step(self):
        self.step_num += 1
        t = self.step_num
        if t <= self.warmup and self.warmup > 0:
            scale = t / float(self.warmup)
        else:
            # cosine from 1 -> 0 over (total - warmup)
            prog = (t - self.warmup) / max(1.0, (self.total - self.warmup))
            scale = 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in self.optimizer.param_groups:
            pg['lr'] = pg.get('_base_lr', pg['lr']) * scale

    def set_base_lrs(self):
        for pg in self.optimizer.param_groups:
            pg['_base_lr'] = pg['lr']


class Agent:
    """Torch MLP agent（CPU-only, reset/train/predict 规范）

    目标：在不启用 GPU 的前提下，争取把 permuted MNIST 单任务准确率推高（≥99% 取决于 epoch/宽度）。
    关键：
      - 特征标准化（/255 后按特征 z-score）
      - 残差 Bottleneck MLP（SiLU + BN + Dropout + WeightNorm）
      - AdamW 默认；也支持 RMSprop 可切换
      - 线性 warmup + 余弦退火（WarmupCosine）
      - 可选动态 INT8 量化（仅推理阶段），默认关闭
    """
    def __init__(
        self,
        output_dim: int = 10,
        seed: int = None,
        input_dim: int = 784,
        width: int = 1024,
        depth: int = 3,
        dropout: float = 0.1,
        batch_size: int = 1024,
        epochs: int = 20,
        lr: float = 2e-3,
        weight_decay: float = 5e-4,
        label_smoothing: float = 0.05,
        optimizer: str = "adamw",   # "adamw" or "rmsprop"
        warmup_steps: int = 200,
        use_dynamic_int8: bool = False,
    ):
        self.output_dim = output_dim
        self.rng = np.random.RandomState(seed)
        self.seed = seed

        # hyperparams
        self.input_dim = input_dim
        self.width = width
        self.depth = depth
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.optimizer_name = optimizer.lower()
        self.warmup_steps = warmup_steps
        self.use_dynamic_int8 = use_dynamic_int8

        # CPU device only
        self.device = torch.device('cpu')

        # runtime
        self.model = None
        self.scaler_mean_ = None
        self.scaler_std_ = None

    def reset(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
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

    def _standardize_fit(self, X: np.ndarray):
        Xf = X.astype(np.float32) / 255.0
        mean = Xf.mean(axis=0)
        std = Xf.std(axis=0)
        std[std < 1e-6] = 1.0
        self.scaler_mean_ = mean
        self.scaler_std_ = std

    def _standardize_transform(self, X: np.ndarray) -> np.ndarray:
        Xf = X.astype(np.float32) / 255.0
        return (Xf - self.scaler_mean_) / self.scaler_std_

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        if self.model is None:
            self.reset()

        X = X_train
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        y = y_train.astype(np.int64).reshape(-1)

        self._standardize_fit(X)
        X = self._standardize_transform(X)

        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        # optimizer
        if self.optimizer_name == "rmsprop":
            opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.9, weight_decay=self.weight_decay, momentum=0.0)
        else:  # adamw
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # scheduler: warmup + cosine over all steps
        total_steps = int(self.epochs * max(1, math.ceil(len(ds) / self.batch_size)))
        sched = WarmupCosine(opt, total_steps=total_steps, warmup_steps=self.warmup_steps)
        sched.set_base_lrs()

        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        self.model.train()
        step = 0
        for epoch in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1

        # optional dynamic INT8 quantization for faster CPU inference
        if self.use_dynamic_int8:
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear}, dtype=torch.qint8
                )
            except Exception:
                # 忽略量化失败（不同 Torch 版本或 Apple Silicon 可能限制）
                pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler_mean_ is None:
            raise RuntimeError("Agent not trained or scaler not fitted.")

        X = X_test
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        X = self._standardize_transform(X)

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


__all__ = ["Agent"]