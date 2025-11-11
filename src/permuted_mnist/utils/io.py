from __future__ import annotations
import pickle
from pathlib import Path

def save_agent(agent, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(agent, f)

def load_agent(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)