"""
PyTorch MLP Agent for Permuted MNIST
Multi-layer perceptron with batch normalization
"""
import torch
from torch import nn
import numpy as np
from time import time
import torch.optim as optim
from copy import deepcopy

class SmoothCrossEntropy(nn.Module):
    """Label-smoothed cross entropy (eps in [0,1))."""
    def __init__(self, eps: float = 0.05, num_classes: int = 10):
        super().__init__()
        self.eps = eps
        self.C = num_classes
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = self.logsoftmax(logits)
        with torch.no_grad():
            t = torch.zeros_like(logp).scatter_(1, targets.view(-1, 1), 1.0)
            t = t * (1.0 - self.eps) + self.eps / self.C
        return (-t * logp).sum(dim=1).mean()

class EpochCosineWithWarmup:
    """Epoch-based warmup + cosine LR schedule."""
    def __init__(self, optimizer: optim.Optimizer, total_epochs: int, warmup_epochs: int = 1):
        self.opt = optimizer
        self.total = max(1, int(total_epochs))
        self.warm = max(1, int(warmup_epochs))
        self.base = [g['lr'] for g in optimizer.param_groups]
        self.ep = 0

    def step(self):
        self.ep += 1
        for i, g in enumerate(self.opt.param_groups):
            base = self.base[i]
            if self.ep <= self.warm:
                lr = base * self.ep / float(self.warm)
            else:
                prog = (self.ep - self.warm) / float(max(1, self.total - self.warm))
                lr = 0.5 * base * (1.0 + np.cos(np.pi * prog))
            g['lr'] = lr

def unit_l2_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample L2 unit normalization on flattened vectors."""
    x_flat = x.view(x.size(0), -1)
    n = x_flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    x_flat = x_flat / n
    return x_flat


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        hidden_sizes = [400, 400]

        layers = []
        d_in = 28 ** 2
        for i, n in enumerate(hidden_sizes):
            layers.append(nn.Linear(d_in, n))
            layers.append(nn.BatchNorm1d(n))
            layers.append(nn.ReLU())
            d_in = n

        layers += [nn.Linear(d_in, 10)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)


class Agent:
    """PyTorch MLP agent for MNIST classification"""

    def __init__(self, output_dim: int = 10, seed: int = None,
                 epochs: int = 20,
                 batch_size: int = 128,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 validation_fraction: float = 0.2,
                 patience: int = 5,
                 use_label_smoothing: bool = True,
                 smoothing_eps: float = 0.05,
                 use_unit_norm: bool = True,
                 optimizer: str = "adam",
                 use_warmup_cosine: bool = True,
                 warmup_epochs: int = 1,
                 preserve_across_tasks: bool = True,
                 verbose: bool = True):

        self.output_dim = output_dim
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.model = Model()
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.validation_fraction = float(validation_fraction)
        self.patience = int(patience)
        self.use_label_smoothing = bool(use_label_smoothing)
        self.smoothing_eps = float(smoothing_eps)
        self.use_unit_norm = bool(use_unit_norm)
        self.optimizer_name = optimizer.lower()
        self.use_warmup_cosine = bool(use_warmup_cosine)
        self.warmup_epochs = int(warmup_epochs)
        self.preserve_across_tasks = bool(preserve_across_tasks)
        self.verbose = bool(verbose)

        # internal trackers
        self._best_state = None
        self._best_val = -np.inf
        self._no_improve = 0

    def reset(self):
        """Reset the agent for a new task/simulation.
        If preserve_across_tasks=True, keep the current model weights (soft reset).
        Otherwise, reinitialize the model (hard reset)."""
        if self.preserve_across_tasks:
            # Soft reset: clear early-stopping trackers only
            self._best_state = None
            self._best_val = -np.inf
            self._no_improve = 0
        else:
            # Hard reset: rebuild model and clear trackers
            self.model = Model()
            self._best_state = None
            self._best_val = -np.inf
            self._no_improve = 0

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the agent on the provided data with validation-based early stopping.
        """
        # Ensure y_train is 1D
        if len(y_train.shape) > 1:
            y_train = y_train.squeeze()

        # Split validation
        N = X_train.shape[0]
        N_val = int(N * self.validation_fraction)
        X_val_np = X_train[:N_val]
        y_val_np = y_train[:N_val]
        X_tr_np = X_train[N_val:]
        y_tr_np = y_train[N_val:]

        # To tensors and scale to [0,1]
        X_tr = torch.from_numpy(X_tr_np).float() / 255.0
        y_tr = torch.from_numpy(y_tr_np).long()
        X_val = torch.from_numpy(X_val_np).float() / 255.0
        y_val = torch.from_numpy(y_val_np).long()

        # Optional per-sample L2 unit norm (operate on flattened views)
        if self.use_unit_norm:
            X_tr = unit_l2_norm(X_tr)
            X_val = unit_l2_norm(X_val)

        # Dataloader
        ds = torch.utils.data.TensorDataset(X_tr, y_tr)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size,
                                             shuffle=True, drop_last=False,
                                             num_workers=0, pin_memory=False)

        # Loss
        if self.use_label_smoothing:
            criterion = SmoothCrossEntropy(self.smoothing_eps, num_classes=self.output_dim)
        else:
            criterion = nn.CrossEntropyLoss()

        # Optimizer
        if self.optimizer_name == "rmsprop":
            opt = optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Scheduler
        scheduler = EpochCosineWithWarmup(opt, total_epochs=self.epochs, warmup_epochs=self.warmup_epochs) if self.use_warmup_cosine else None

        # Reset early stopping trackers
        self._best_state = deepcopy(self.model.state_dict())
        self._best_val = -np.inf
        self._no_improve = 0

        # Training loop
        for ep in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

            if scheduler is not None:
                scheduler.step()

            # Validation accuracy
            self.model.eval()
            with torch.no_grad():
                logits_val = self.model(X_val)
                pred_val = torch.argmax(logits_val, dim=1)
                acc_val = (pred_val == y_val).float().mean().item()

            if self.verbose and self.validation_fraction > 0:
                print(f"epoch {ep}: val_acc = {acc_val*100:.2f}%")

            # Early stopping logic
            if acc_val > self._best_val + 1e-6:
                self._best_val = acc_val
                self._best_state = deepcopy(self.model.state_dict())
                self._no_improve = 0
            else:
                self._no_improve += 1
                if self._no_improve >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {ep} (best val_acc={self._best_val*100:.2f}%).")
                    break

        # Load best weights
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions on test data."""
        if isinstance(X_test, np.ndarray):
            X = torch.from_numpy(X_test).float() / 255.0
        else:
            X = X_test.float()
        if self.use_unit_norm:
            X = unit_l2_norm(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model.forward(X)
        return logits.argmax(-1).detach().cpu().numpy()


if __name__ == "__main__":
    agent = Agent()
    X_train = np.load("./data/mnist_train_images.npy")
    y_train = np.load("./data/mnist_train_labels.npy")
    X_test = np.load("./data/mnist_test_images.npy")
    y_test = np.load("./data/mnist_test_labels.npy")

    t0 = time()
    agent.train(X_train, y_train)
    y_predict = agent.predict(X_test)
    is_correct = y_predict == y_test
    acc = np.mean(is_correct)
    print(f"Test accuracy: {acc:0.04f} in {time() - t0:.2f} seconds")
