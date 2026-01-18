
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two identical high-capacity sine networks on 1D regression with large white noise:
  - Baseline: standard MSE
  - Ours: MDL-style batch surrogate (natural logs only)

Hidden layers: sine activation
Output layer: identity (linear)

Quick knobs (MACROS) at the top and/or CLI flags:
  --num-hidden, --hidden-dim, --epochs, --batch-size, --lr, --curv-m, --prior-lambda, --noise-std

Dependencies: torch, numpy, matplotlib
"""

import os
import math
import random
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ========== MACROS (edit here or override via CLI) ==========

SEED: int = 1234
N_SAMPLES: int = 500        # total dataset size (n)
BATCH_SIZE: int = 10        # batch size (B)
EPOCHS: int = 1000
LEARNING_RATE: float = 1e-3
PRIOR_LAMBDA: float = 0.0    # multiplies the prior term
CURV_CADENCE_M: int = 1      # compute curvature every m steps (1 = every step)
CLIP_MAX_NORM: float = 5.0

# Model size macros:
MODEL_NUM_HIDDEN: int = 5    # <--- number of hidden layers
MODEL_HIDDEN_DIM: int = 256  # <--- width of each hidden layer

# Signal / noise
NOISE_STD: float = 0.1
W1: float = 3.0
P1: float = 0.25 * math.pi
W2: float = 7.0
P2: float = 0.75 * math.pi
A2: float = 0.6
X_MIN: float = 0.0
X_MAX: float = 2.0 * math.pi

# Numerics
EPS_MSE: float = 1e-12
EPS_W2: float = 1e-32

# ============================================================


@dataclass
class Config:
    seed: int = SEED
    n_samples: int = N_SAMPLES
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    lr: float = LEARNING_RATE
    prior_lambda: float = PRIOR_LAMBDA
    curv_m: int = CURV_CADENCE_M
    clip_max_norm: float = CLIP_MAX_NORM
    num_hidden: int = MODEL_NUM_HIDDEN
    hidden_dim: int = MODEL_HIDDEN_DIM
    noise_std: float = NOISE_STD
    w1: float = W1
    p1: float = P1
    w2: float = W2
    p2: float = P2
    a2: float = A2
    x_min: float = X_MIN
    x_max: float = X_MAX


def parse_args_to_config() -> Config:
    p = argparse.ArgumentParser(description="MDL vs MSE on sine+noise with sine MLP")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--prior-lambda", type=float, default=PRIOR_LAMBDA)
    p.add_argument("--curv-m", type=int, default=CURV_CADENCE_M,
                   help="compute curvature every m steps (1 = every step)")
    p.add_argument("--clip-max-norm", type=float, default=CLIP_MAX_NORM)
    p.add_argument("--num-hidden", type=int, default=MODEL_NUM_HIDDEN,
                   help="NUMBER OF HIDDEN LAYERS (sine)")
    p.add_argument("--hidden-dim", type=int, default=MODEL_HIDDEN_DIM,
                   help="WIDTH of each hidden layer")
    p.add_argument("--noise-std", type=float, default=NOISE_STD)
    p.add_argument("--w1", type=float, default=W1)
    p.add_argument("--p1", type=float, default=P1)
    p.add_argument("--w2", type=float, default=W2)
    p.add_argument("--p2", type=float, default=P2)
    p.add_argument("--a2", type=float, default=A2)
    p.add_argument("--x-min", type=float, default=X_MIN)
    p.add_argument("--x-max", type=float, default=X_MAX)
    args = p.parse_args()

    return Config(
        seed=args.seed,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        prior_lambda=args.prior_lambda,
        curv_m=args.curv_m,
        clip_max_norm=args.clip_max_norm,
        num_hidden=args.num_hidden,
        hidden_dim=args.hidden_dim,
        noise_std=args.noise_std,
        w1=args.w1, p1=args.p1, w2=args.w2, p2=args.p2, a2=args.a2,
        x_min=args.x_min, x_max=args.x_max
    )


def setup_seed_and_device(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def make_dataset(cfg: Config):
    x = np.random.uniform(cfg.x_min, cfg.x_max, size=(cfg.n_samples, 1)).astype(np.float32)
    y_clean = np.sin(cfg.w1 * x + cfg.p1) + cfg.a2 * np.sin(cfg.w2 * x + cfg.p2)
    noise = cfg.noise_std * np.random.randn(cfg.n_samples, 1).astype(np.float32)
    y = y_clean + noise
    return x, y, y_clean


class SineMLP(nn.Module):
    """Hidden layers: sine activation; output layer: identity."""
    def __init__(self, in_dim=1, out_dim=1, hidden_dim=256, num_hidden=5):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(last, hidden_dim))
            last = hidden_dim
        self.fcs = nn.ModuleList(layers)
        self.out = nn.Linear(last, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # Small-variance init to avoid immediate high-frequency regimes
        for m in self.fcs:
            nn.init.uniform_(m.weight, a=-0.5 / m.in_features, b=0.5 / m.in_features)
            nn.init.zeros_(m.bias)
        nn.init.uniform_(self.out.weight, a=-0.5 / self.out.in_features, b=0.5 / self.out.in_features)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = x
        for m in self.fcs:
            h = torch.sin(m(h))
        return self.out(h)  # identity at output


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mean_w2_graph(model: nn.Module, device) -> torch.Tensor:
    """Mean of squared parameters with gradient."""
    sq_sum = torch.zeros((), device=device)
    n_params = 0
    for p in model.parameters():
        if p.requires_grad:
            sq_sum = sq_sum + p.pow(2).sum()
            n_params += p.numel()
    return (sq_sum / max(1, n_params)).clamp_min(EPS_W2)


def curvature_term_and_gnorm2(model: nn.Module, xb: torch.Tensor):
    """
    Return:
      curv_term = ln(1 + 2 * ||g_hat||^2)  (scalar, with graph)
      g_norm2   = ||g_hat||^2 (detached, for logging)
    where g_hat = average gradient (over the batch) of the scalar output wrt parameters.
    """
    out = model(xb)               # [B, 1]
    mean_out = out.mean()         # scalar
    grads = torch.autograd.grad(
        mean_out, [p for p in model.parameters() if p.requires_grad],
        create_graph=True, retain_graph=True, allow_unused=False
    )
    g_sq = torch.zeros((), device=xb.device)
    for g in grads:
        g_sq = g_sq + (g.reshape(-1) @ g.reshape(-1))
    curv_term = torch.log(1.0 + 2.0 * g_sq)  # natural log
    return curv_term, g_sq.detach()


def build_loader(x, y, cfg: Config, device):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    use_cuda = (device.type == "cuda")
    num_workers = min(4, os.cpu_count() or 1)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        pin_memory=use_cuda, num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )
    return loader


def train_baseline_mse(model, loader, n_data, cfg: Config, device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    hist = []
    for ep in range(1, cfg.epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            mse_b = F.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            mse_b.backward()
            opt.step()
            ep_loss += mse_b.item()
        hist.append(ep_loss / len(loader))
    return hist


def train_mdl(model, loader, n_data, cfg: Config, device):
    """
    Our per-batch loss (natural logs only):
      L = B * ln(MSE_batch + eps)
        + (B * ln(n) / n) * ln(1 + 2 * ||g_hat||^2)
        + (B / n) * prior_lambda * |omega| * ln(mean(w^2))
    """
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    n_params = count_parameters(model)

    losses_total, losses_mse, losses_curv, losses_prior = [], [], [], []
    curv_proxy_vals, gnorm2_vals = [], []

    step = 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        ep_total = ep_mse = ep_curv = ep_prior = ep_curv_proxy = ep_gn2 = 0.0

        for xb, yb in loader:
            step += 1
            B = xb.shape[0]
            xb = xb.to(device); yb = yb.to(device)

            pred = model(xb)
            mse_b = F.mse_loss(pred, yb)

            # Curvature cadence:
            if cfg.curv_m <= 1 or (step % cfg.curv_m) == 0:
                curv_term, g_norm2 = curvature_term_and_gnorm2(model, xb)
                curv_loss = (B * math.log(n_data) / n_data) * curv_term
                ep_curv_proxy += curv_term.item()
                ep_gn2 += g_norm2.item()
            else:
                # Skip curvature this step (0 contribution to gradient)
                curv_loss = torch.zeros((), device=device)

            # Prior with gradient:
            mw2 = mean_w2_graph(model, device)
            prior_loss = (B / n_data) * cfg.prior_lambda * n_params * torch.log(mw2)

            loss = B * torch.log(mse_b + EPS_MSE) + curv_loss + prior_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_max_norm)
            opt.step()

            ep_total += loss.item()
            ep_mse   += (B * torch.log(mse_b + EPS_MSE)).item()
            ep_curv  += curv_loss.item()
            ep_prior += prior_loss.item()

        denom = len(loader)
        losses_total.append(ep_total / denom)
        losses_mse.append(ep_mse / denom)
        losses_curv.append(ep_curv / denom)
        losses_prior.append(ep_prior / denom)
        if cfg.curv_m <= 1:
            curv_proxy_vals.append(ep_curv_proxy / denom)
            gnorm2_vals.append(ep_gn2 / denom)
        else:
            # proxy only updated on refreshed steps; still log the average per epoch
            curv_proxy_vals.append(ep_curv_proxy / max(1, (denom // max(1, cfg.curv_m))))
            gnorm2_vals.append(ep_gn2 / max(1, (denom // max(1, cfg.curv_m))))

    return {
        "loss_total": losses_total,
        "loss_mse": losses_mse,
        "loss_curv": losses_curv,
        "loss_prior": losses_prior,
        "curv_proxy": curv_proxy_vals,
        "g_norm2": gnorm2_vals,
    }


def main():
    cfg = parse_args_to_config()
    device = setup_seed_and_device(cfg)

    print(f"Device       : {device}")
    print(f"Model (sin)  : hidden_dim={cfg.hidden_dim}, num_hidden={cfg.num_hidden}")
    print(f"Data         : n={cfg.n_samples}, batch={cfg.batch_size}, noise_std={cfg.noise_std}")
    print(f"Training     : epochs={cfg.epochs}, lr={cfg.lr}, curv_m={cfg.curv_m}, prior_lambda={cfg.prior_lambda}")

    # Data
    x, y, y_clean = make_dataset(cfg)
    loader = build_loader(x, y, cfg, device)

    # Models
    model_mse = SineMLP(in_dim=1, out_dim=1,
                        hidden_dim=cfg.hidden_dim, num_hidden=cfg.num_hidden)
    model_mdl = SineMLP(in_dim=1, out_dim=1,
                        hidden_dim=cfg.hidden_dim, num_hidden=cfg.num_hidden)

    # Train
    print("Training baseline (MSE)...")
    mse_hist = train_baseline_mse(model_mse, loader, cfg.n_samples, cfg, device)

    print("Training ours (MDL batch surrogate)...")
    mdl_hist = train_mdl(model_mdl, loader, cfg.n_samples, cfg, device)

    # Eval grid
    with torch.no_grad():
        x_grid = np.linspace(cfg.x_min, cfg.x_max, 2000, dtype=np.float32).reshape(-1, 1)
        xg_t = torch.from_numpy(x_grid).to(device)
        y_clean_grid = np.sin(cfg.w1 * x_grid + cfg.p1) + cfg.a2 * np.sin(cfg.w2 * x_grid + cfg.p2)

        model_mse.eval(); model_mdl.eval()
        y_mse = model_mse(xg_t).cpu().numpy()
        y_mdl = model_mdl(xg_t).cpu().numpy()

    # Plots
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    ax1.plot(mse_hist, label="Baseline (MSE)", color="#1f77b4", lw=2)
    ax1.plot(mdl_hist["loss_total"], label="Ours (total)", color="#d62728", lw=2)
    ax1.plot(mdl_hist["loss_mse"], label="Ours data term", color="#2ca02c", ls="--")
    ax1.plot(mdl_hist["loss_curv"], label="Ours curvature term", color="#9467bd", ls=":")
    ax1.plot(mdl_hist["loss_prior"], label="Ours prior term", color="#8c564b", ls="-.")
    ax1.set_title("Training curves (natural logs)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 5))
    # visualize a subset of noisy points
    idx_vis = np.random.choice(len(x), size=min(1200, len(x)), replace=False)
    ax2.scatter(x[idx_vis, 0], y[idx_vis, 0], s=6, alpha=0.15, label="Noisy samples", color="gray")
    ax2.plot(x_grid[:, 0], y_clean_grid[:, 0], color="black", lw=2.0, alpha=0.9, label="Clean target")
    ax2.plot(x_grid[:, 0], y_mse[:, 0], color="#1f77b4", lw=2.0, label="Baseline (MSE)")
    ax2.plot(x_grid[:, 0], y_mdl[:, 0], color="#d62728", lw=2.0, label="Ours (MDL loss)")
    ax2.set_title("Fit comparison: Sine MLP (hidden: sin, output: identity)")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    ax3.plot(mdl_hist["curv_proxy"], color="#9467bd", lw=2, label="ln(1 + 2||ĝ||²)")
    ax3.set_title("Curvature proxy (ours)"); ax3.set_xlabel("Epoch"); ax3.set_ylabel("Value")
    ax3.grid(True, alpha=0.3); ax3.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
