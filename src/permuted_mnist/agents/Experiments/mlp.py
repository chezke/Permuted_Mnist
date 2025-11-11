# ===== TorchMLP (RMSprop + FE + SiLU + Bottleneck + WeightNorm + LabelSmoothing + WarmupCosine + Dynamic INT8) =====
import time, numpy as np
import torch, torch.nn as nn, torch.optim as optim

try:
    torch.set_num_threads(min(2, torch.get_num_threads()))
except Exception:
    pass

def _as_float_01(x_np):
    x = torch.from_numpy(x_np).float()
    if x.max() > 1.0:
        x = x / 255.0
    return x

def _labels_1d(y_np):
    y = np.asarray(y_np).reshape(-1).astype(np.int64, copy=False)
    return torch.from_numpy(y)

def _l2_per_sample(x, eps=1e-6):
    orig = x.shape
    if x.dim() == 3:
        x = x.view(x.size(0), -1)
    n = x.norm(dim=1, keepdim=True).clamp_min(eps)
    x = x / n
    if len(orig) == 3:
        x = x.view(orig)
    return x

# ------------------------- 损失与调度 -------------------------
class SmoothCE(nn.Module):
    def __init__(self, eps=0.05, num_classes=10):
        super().__init__()
        self.eps = eps
        self.C = num_classes
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, target):
        logp = self.logsoftmax(logits)
        with torch.no_grad():
            t = torch.zeros_like(logp).scatter_(1, target.view(-1, 1), 1.0)
            t = t * (1 - self.eps) + self.eps / self.C
        return (-t * logp).sum(dim=1).mean()

class WarmupCosine:
    def __init__(self, optimizer, total_epochs, warmup_epochs=1):
        self.opt = optimizer
        self.total = total_epochs
        self.warm = warmup_epochs
        self.base = [g['lr'] for g in optimizer.param_groups]
        self.ep = 0

    def step(self):
        self.ep += 1
        for i, g in enumerate(self.opt.param_groups):
            base = self.base[i]
            if self.ep <= self.warm:
                lr = base * self.ep / max(1, self.warm)
            else:
                prog = (self.ep - self.warm) / max(1, (self.total - self.warm))
                lr = 0.5 * base * (1 + np.cos(np.pi * prog))
            g['lr'] = lr

# ------------------------- 模型 -------------------------
class Bottleneck(nn.Module):
    def __init__(self, d=256, b=64):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(d)
        self.act1 = nn.SiLU(inplace=True)
        self.fc1 = nn.Linear(d, b, bias=False)
        self.bn2 = nn.BatchNorm1d(b)
        self.act2 = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(b, d, bias=False)
        nn.init.kaiming_uniform_(self.fc1.weight, a=0, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, a=0, nonlinearity='relu')

    def forward(self, x):
        h = self.fc1(self.act1(self.bn1(x)))
        h = self.fc2(self.act2(self.bn2(h)))
        return x + h

class _MLP_ResBN(nn.Module):
    def __init__(self, in_dim=784, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = nn.SiLU(inplace=True)
        self.block = Bottleneck(d=256, b=64)
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.ReLU(inplace=True)
        head = nn.Linear(128, out_dim, bias=True)
        self.head = nn.utils.weight_norm(head)
        for m in [self.fc1, self.fc2, head]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.block(x)
        x = self.act2(self.bn2(self.fc2(x)))
        return self.head(x)

# ------------------------- Agent（ dynamic INT8 compress） -------------------------
class TorchMLP:
    def __init__(self, output_dim=10, seed=42, epochs=10, batch_size=128, lr=1e-3,
                 val_ratio=0.2, weight_decay=0.0):
        self.output_dim = output_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_ratio = val_ratio
        self.weight_decay = weight_decay
        self.device = torch.device("cpu")
        np.random.seed(seed); torch.manual_seed(seed)
        self.model = None
        self.model_int8 = None
        self.reset()

    def reset(self):
        self.model = _MLP_ResBN(in_dim=784, out_dim=self.output_dim).to(self.device)
        self.model_int8 = None

    def _make_loader(self, X, y, shuffle=True):
        ds = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            drop_last=False, num_workers=0, pin_memory=False
        )

    def train(self, X_train, y_train):
        X = _as_float_01(X_train); y = _labels_1d(y_train)
        n_total = X.shape[0]; n_val = int(self.val_ratio * n_total)
        idx = np.arange(n_total); np.random.shuffle(idx)
        val_idx = idx[:n_val]; tr_idx = idx[n_val:]
        X_tr, y_tr = X[tr_idx].to(self.device), y[tr_idx].to(self.device)

        X_tr = _l2_per_sample(X_tr)
        loader = self._make_loader(X_tr, y_tr, shuffle=True)

        opt = optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.99,
                            momentum=0.0, centered=False, weight_decay=self.weight_decay)
        sch = WarmupCosine(opt, total_epochs=self.epochs, warmup_epochs=1)
        crit = SmoothCE(0.05, self.output_dim)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
            sch.step()

    def compress_dynamic_int8(self):
        import torch.quantization as tq
        mdl = self.model.to("cpu").eval()
        try:
            nn.utils.remove_weight_norm(mdl.head)
        except Exception:
            pass
        self.model_int8 = tq.quantize_dynamic(
            mdl, {nn.Linear}, dtype=torch.qint8, inplace=False
        )

    @torch.no_grad()
    def predict(self, X_test):
        X = _as_float_01(X_test)
        X = _l2_per_sample(X)
        mdl = self.model_int8 if self.model_int8 is not None else self.model
        mdl.eval()
        bs = 4096
        outs = []
        for i in range(0, X.shape[0], bs):
            logits = mdl(X[i:i+bs])
            outs.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(outs, axis=0)