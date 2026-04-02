import copy
import math
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _GateClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        temperature: float = 0.66,
        gamma: float = -0.1,
        zeta: float = 1.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.log_alpha = nn.Parameter(torch.zeros(self.input_dim))
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _sample_hard_concrete(self) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp_(1e-6, 1.0 - 1e-6)
            s = torch.sigmoid(
                (torch.log(u) - torch.log1p(-u) + self.log_alpha) / self.temperature
            )
        else:
            s = torch.sigmoid(self.log_alpha)
        stretched = s * (self.zeta - self.gamma) + self.gamma
        return torch.clamp(stretched, 0.0, 1.0)

    def expected_gate_probabilities(self) -> torch.Tensor:
        offset = self.temperature * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha - offset)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self._sample_hard_concrete()
        logits = self.classifier(x * gates).squeeze(-1)
        return logits, gates


class StochasticGateFeatureSelector:
    """
    Supervised feature selector based on stochastic gates.

    The selector is fit on pooled 768-d sequence embeddings and returns a
    shared set of hidden dimensions to keep for all token embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int = 300,
        hidden_dim: int = 256,
        max_epochs: int = 40,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        temperature: float = 0.66,
        lambda_target: float = 5e-4,
        lambda_l0: float = 1e-4,
        random_state: int = 42,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.input_dim = int(input_dim)
        self.target_dim = int(target_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)
        self.lambda_target = float(lambda_target)
        self.lambda_l0 = float(lambda_l0)
        self.random_state = int(random_state)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = bool(verbose)

        self.selected_idx_: np.ndarray | None = None
        self.gate_probabilities_: np.ndarray | None = None
        self.history_: list[dict[str, float]] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "StochasticGateFeatureSelector":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y).astype(np.float32).ravel()
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape={x.shape}")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"input_dim mismatch: expected {self.input_dim}, got {x.shape[1]}"
            )
        if self.target_dim <= 0:
            raise ValueError("target_dim must be > 0")

        set_torch_seed(self.random_state)

        dataset = TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            drop_last=False,
        )

        model = _GateClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            temperature=self.temperature,
        ).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        best_state = None
        best_loss = float("inf")

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_cls = 0.0
            epoch_gate = 0.0
            epoch_l0 = 0.0
            seen = 0

            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(xb)
                cls_loss = criterion(logits, yb)

                gate_probs = model.expected_gate_probabilities()
                target_penalty = ((gate_probs.sum() - self.target_dim) ** 2) / self.input_dim
                l0_penalty = gate_probs.mean()

                loss = cls_loss + self.lambda_target * target_penalty + self.lambda_l0 * l0_penalty
                loss.backward()
                optimizer.step()

                batch_n = xb.shape[0]
                seen += batch_n
                epoch_loss += float(loss.item()) * batch_n
                epoch_cls += float(cls_loss.item()) * batch_n
                epoch_gate += float(target_penalty.item()) * batch_n
                epoch_l0 += float(l0_penalty.item()) * batch_n

            mean_loss = epoch_loss / max(seen, 1)
            mean_cls = epoch_cls / max(seen, 1)
            mean_gate = epoch_gate / max(seen, 1)
            mean_l0 = epoch_l0 / max(seen, 1)
            self.history_.append(
                {
                    "epoch": float(epoch),
                    "loss": mean_loss,
                    "cls_loss": mean_cls,
                    "target_penalty": mean_gate,
                    "l0_penalty": mean_l0,
                }
            )
            if self.verbose:
                print(
                    f"[SG] epoch={epoch:02d} loss={mean_loss:.6f} "
                    f"cls={mean_cls:.6f} target_penalty={mean_gate:.6f} l0={mean_l0:.6f}"
                )

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError("Stochastic gate training did not produce a valid state.")

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            gate_probs = model.expected_gate_probabilities().detach().cpu().numpy()

        keep = min(self.target_dim, self.input_dim)
        selected_idx = np.argsort(-gate_probs)[:keep]
        selected_idx = np.sort(selected_idx.astype(int))

        self.selected_idx_ = selected_idx
        self.gate_probabilities_ = gate_probs.astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.selected_idx_ is None:
            raise RuntimeError("The selector must be fit before calling transform().")
        arr = np.asarray(x)
        return arr[..., self.selected_idx_]

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(x, y).transform(x)
