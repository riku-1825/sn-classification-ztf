from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class LightCurveDataset(Dataset):
    def __init__(self, X: np.ndarray, mask: np.ndarray, y: np.ndarray):
        n, max_len, n_bands, n_feat = X.shape
        self.X = X.reshape(n, max_len, n_bands * n_feat).astype(np.float32)
        self.mask = (mask.sum(axis=2) > 0).astype(np.float32)  # [N, max_len]
        self.lengths = self.mask.sum(axis=1).clip(min=1).astype(np.int64)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.lengths[idx]),
            torch.tensor(self.y[idx]),
        )


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1,
                 num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        last_hidden = h_n[-1]  # [batch, hidden_dim]
        return self.head(last_hidden)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_gru(model, train_loader, val_loader, epochs=30, lr=3e-3, device=None, patience=8):
    device = device or get_device()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        for xb, lens, yb in train_loader:
            xb, lens, yb = xb.to(device), lens.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb, lens)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for xb, lens, yb in val_loader:
                xb, lens, yb = xb.to(device), lens.to(device), yb.to(device)
                out = model(xb, lens)
                val_loss += loss_fn(out, yb).item() * len(yb)
                n += len(yb)
        val_loss /= max(n, 1)
        print(f"epoch {epoch+1:02d} | val_loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_gru(model, loader, device=None):
    device = device or get_device()
    model.to(device).eval()
    all_probs, all_labels = [], []
    for xb, lens, yb in loader:
        xb, lens = xb.to(device), lens.to(device)
        logits = model(xb, lens)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(yb.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
