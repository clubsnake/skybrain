"""
Training script for the navigation Liquid Neural Network (NavLNN).

This script defines a simple model using the LTCCell from `ltc_cell.py` and an output
head to map the hidden state to the desired navigation caps (forward, lateral,
vertical velocities, yaw rate cap, scan probability, ramp gain, confidence).

The training data should be prepared as sequences of feature vectors and target
outputs. See `docs/contracts/policy_io.md` for the exact feature order and
output semantics. Datasets are expected to be stored in a directory of
NumPy `.npz` files with arrays `features` and `targets` of equal length.

Example usage:

```
python train_navlnn.py \
    --data-dir ./data/navlnn \
    --epochs 50 \
    --hidden-dim 32 \
    --lr 1e-3 \
    --out-model nav_lnn.tflite
```

This script is provided as a starting point and does not perform any
validation, early stopping, or model export. You should adapt it to your
dataset and training workflow.
"""
import argparse
import os
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    # If run from training/ directory
    from pytorch.ltc_cell import LTCCell  # type: ignore
except Exception:  # pragma: no cover
    # Fallback when executed from repo root
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'pytorch'))
    from ltc_cell import LTCCell


class NavLNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 7):
        super().__init__()
        self.rnn = LTCCell(input_dim, hidden_dim)
        # Fully connected head maps hidden state + input to outputs
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim)
        )

    def forward(self, u: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one timestep.

        Args:
            u: Input features at current timestep (batch, input_dim)
            h: Hidden state from previous timestep (batch, hidden_dim)

        Returns:
            Tuple of (outputs, next_hidden_state)
        """
        next_h, _tau = self.rnn(u, h)
        head_in = torch.cat([next_h, u], dim=-1)
        out = self.head(head_in)
        return out, next_h


def load_dataset(data_dir: Path) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Iterates over dataset files in `data_dir`. Each file should be a `.npz` with
    arrays `features` (shape [T, F]) and `targets` (shape [T, O])."""
    for file in sorted(data_dir.glob("*.npz")):
        data = np.load(file)
        feats = data["features"]
        targs = data["targets"]
        yield feats, targs


def train(model: NavLNN, data_iter: Iterator[Tuple[np.ndarray, np.ndarray]], epochs: int, lr: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for feats_np, targs_np in data_iter:
            feats = torch.tensor(feats_np, dtype=torch.float32, device=device)
            targs = torch.tensor(targs_np, dtype=torch.float32, device=device)
            # Initialize hidden state to zeros for each sequence
            h = torch.zeros((1, model.rnn.W_in.out_features), device=device)
            for t in range(feats.shape[0]):
                u_t = feats[t : t + 1]
                target_t = targs[t : t + 1]
                pred_t, h = model(u_t, h)
                loss = criterion(pred_t, target_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1
        print(f"Epoch {epoch+1}/{epochs}: avg loss={total_loss / max(count,1):.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train the NavLNN model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-model", type=str, default="nav_lnn.pth")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    # Determine input and output dimensions from first file
    first_file = next(data_dir.glob("*.npz"), None)
    if first_file is None:
        raise RuntimeError(f"No .npz files found in {data_dir}")
    sample = np.load(first_file)
    input_dim = sample["features"].shape[1]
    output_dim = sample["targets"].shape[1]
    model = NavLNN(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim)

    train(model, load_dataset(data_dir), epochs=args.epochs, lr=args.lr)

    # Save PyTorch model
    torch.save(model.state_dict(), args.out_model)
    print(f"Model saved to {args.out_model}")


if __name__ == "__main__":
    main()