# agent/my_mlp/agent.py
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

def _seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class _Block(nn.Module):
    def __init__(self, in_f, out_f, p_drop=0.2):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_f, out_f),
            nn.BatchNorm1d(out_f),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )
    def forward(self, x):
        return self.seq(x)

class _MLP(nn.Module):
    def __init__(self, in_f=784, n_classes=10, width=1024, depth=3, p_drop=0.2):
        super().__init__()
        layers = []
        hidden = width
        layers.append(_Block(in_f, hidden, p_drop))
        for _ in range(depth - 1):
            layers.append(_Block(hidden, hidden, p_drop))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_classes)
        # Kaiming init for ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

class Agent:
    """
    Torch MLP agent for Permuted MNIST
    - 三层宽 MLP (ReLU+BN+Dropout)
    - AdamW + 余弦退火学习率
    - Label smoothing CrossEntropy
    - 每任务标准化（保存 mean/std, 预测用同一套）
    - 早停 + 梯度裁剪
    """
    def __init__(
        self,
        output_dim: int = 10,
        seed: int = 42,
        input_dim: int = 784,
        width: int = 1024,
        depth: int = 3,
        dropout: float = 0.2,
        batch_size: int = 1024,
        epochs: int = 18,                # 如未达 99%，可提到 22-25
        lr: float = 3e-3,
        weight_decay: float = 5e-4,
        label_smoothing: float = 0.05,
        val_ratio: float = 0.1,
        early_stop_patience: int = 5,
        grad_clip_norm: float = 1.0,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_seconds_per_task: float = 5.0,
    ):
        _seed_everything(seed)
        self.seed = seed
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.width = width
        self.depth = depth
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.val_ratio = val_ratio
        self.early_stop_patience = early_stop_patience
        self.grad_clip_norm = grad_clip_norm
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_seconds_per_task = max_seconds_per_task

        self.device = torch.device("cpu")
        self.model = None
        self.scaler_mean_ = None
        self.scaler_std_ = None

    # 你的评测脚本每个任务都会 reset()
    def reset(self):
        _seed_everything(self.seed)
        self.model = _MLP(in_f=self.input_dim, n_classes=self.output_dim,
                          width=self.width, depth=self.depth, p_drop=self.dropout).to(self.device)
        self.scaler_mean_ = None
        self.scaler_std_ = None

    def _fit_standardizer(self, X: np.ndarray):
        # 输入通常是0..255；先/255，再按特征做标准化
        Xf = X.astype(np.float32) / 255.0
        mean = Xf.mean(axis=0)
        std = Xf.std(axis=0)
        std[std < 1e-6] = 1.0  # 防止除零（对几乎恒定像素）
        self.scaler_mean_ = mean
        self.scaler_std_ = std

    def _transform(self, X: np.ndarray) -> np.ndarray:
        Xf = X.astype(np.float32) / 255.0
        if self.scaler_mean_ is None or self.scaler_std_ is None:
            raise RuntimeError("Standardizer not fitted yet.")
        return (Xf - self.scaler_mean_) / self.scaler_std_

    def _make_loaders(self, X: np.ndarray, y: np.ndarray):
        # 拆分训练/验证
        N = X.shape[0]
        val_size = max(1, int(self.val_ratio * N))
        train_size = N - val_size
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y.astype(np.int64))
        ds = TensorDataset(X_t, y_t)
        gen = torch.Generator().manual_seed(self.seed)
        train_ds, val_ds = random_split(ds, [train_size, val_size], generator=gen)

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        return train_loader, val_loader

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            logits = self.model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
        return correct / max(1, total)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        if self.model is None:
            self.reset()

        start_t = time.time()

        # 标准化
        self._fit_standardizer(X_train)
        X_tr = self._transform(X_train)

        # DataLoader
        train_loader, val_loader = self._make_loaders(X_tr, y_train)

        # 优化器 + 余弦退火（含 warmup 的简化版：前 1 epoch 降 lr *0.3）
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = self.epochs * math.ceil(len(train_loader))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        best_val = 0.0
        best_state = None
        patience = 0
        step = 0

        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                # 简单 warmup：第一轮把 lr 暂降
                if step < len(train_loader):
                    for pg in opt.param_groups:
                        pg['lr'] = self.lr * 0.3

                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                opt.step()
                scheduler.step()
                step += 1

                # 训练时间限制（CPU 环境更稳）
                if (time.time() - start_t) > self.max_seconds_per_task:
                    break
            if (time.time() - start_t) > self.max_seconds_per_task:
                break

            # 每轮结束做一次验证早停
            val_acc = self._evaluate(val_loader)
            if val_acc > best_val:
                best_val = val_acc
                patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    @torch.no_grad()
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_te = self._transform(X_test)
        ds = TensorDataset(torch.from_numpy(X_te))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)
        preds = []
        for (xb,) in loader:
            xb = xb.to(self.device, non_blocking=True)
            logits = self.model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)