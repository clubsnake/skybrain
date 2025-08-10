"""
Training script for the cinematic Liquid Neural Network (CineLNN).

The CineLNN is responsible for camera-related behaviors such as exposure bias,
gimbal ramp smoothing, bracketing cadence, and other artistic parameters. This
script defines a model similar to NavLNN but with outputs tailored for
camera/gimbal control. The input features may include scene class labels
(produced by the classifier), current camera settings, mission progress, and
other context.

Example usage:

```
python train_cinelnn.py \
    --data-dir ./data/cinelnn \
    --epochs 50 \
    --hidden-dim 32 \
    --lr 1e-3 \
    --out-model cine_lnn.pth
```

As with NavLNN, this script is a template. You should adapt it to your own
dataset format, loss functions, and evaluation criteria.
"""
import argparse
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from pytorch.ltc_cell import LTCCell  # type: ignore
except Exception:  # pragma: no cover
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'pytorch'))
    from ltc_cell import LTCCell


class CineLNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 4):
        super().__init__()
        self.rnn = LTCCell(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim)
        )

    def forward(self, u: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        next_h, _ = self.rnn(u, h)
        head_in = torch.cat([next_h, u], dim=-1)
        out = self.head(head_in)
        return out, next_h


def load_dataset(data_dir: Path) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    for file in sorted(data_dir.glob("*.npz")):
        data = np.load(file)
        yield data["features"], data["targets"]


def train(model: CineLNN, data_iter: Iterator[Tuple[np.ndarray, np.ndarray]], epochs: int, lr: float):
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
    parser = argparse.ArgumentParser(description="Train the CineLNN model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-model", type=str, default="cine_lnn.pth")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    first_file = next(data_dir.glob("*.npz"), None)
    if first_file is None:
        raise RuntimeError(f"No .npz files found in {data_dir}")
    sample = np.load(first_file)
    input_dim = sample["features"].shape[1]
    output_dim = sample["targets"].shape[1]
    model = CineLNN(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim)

    train(model, load_dataset(data_dir), epochs=args.epochs, lr=args.lr)

    torch.save(model.state_dict(), args.out_model)
    print(f"CineLNN saved to {args.out_model}")


if __name__ == "__main__":
    main()