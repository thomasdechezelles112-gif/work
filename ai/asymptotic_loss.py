import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

torch.manual_seed(0)
np.random.seed(0)

N = 512
BATCH_SIZE = 32
EPOCHS = 200
LR = 5e-3

C_GAP = 10.0  # spacing multiplier for sparse LS samples

# ============================================================
# HIGHLY IRREGULAR DATA
# ============================================================

x = torch.linspace(-3, 3, N).unsqueeze(1)
y = torch.sin(3*x) + 0.3*torch.randn_like(x)

# ============================================================
# MODEL
# ============================================================

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32,1)
        )

    def forward(self, x):
        return self.net(x)

model = Net()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
loss_fn = nn.MSELoss()

# ============================================================
# BETA ESTIMATOR AND VARIANCE
# ============================================================

gamma = 0.05
k = 10.0
eps = 1e-12

v = 0.0      # variance of r_n (×2)
betahat = 0.0
v2 = 0.0     # variance of betahat updates (×2)

loss_hist = []
beta_hist = []

# ============================================================
# SPARSE SAMPLING STORAGE
# ============================================================

sample_n = []
sample_L = []
last_sample_n = 0

# ============================================================
# TRAINING LOOP
# ============================================================

step = 0
for epoch in range(EPOCHS):
    perm = torch.randperm(N)

    for i in range(0, N, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        xb, yb = x[idx], y[idx]

        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        L = loss.item()
        loss_hist.append(L)
        beta_hist.append(betahat)
        step += 1

        # ---------- update beta ----------
        if step >= 4:
            L0, L1, L2, L3 = loss_hist[-4:]
            d1, d2, d3 = L1-L0, L2-L1, L3-L2

            if abs(d1) > eps and abs(d2) > eps and abs(d3) > eps:
                r_n   = np.log(abs(d1 / d2))
                r_np1 = np.log(abs(d2 / d3))

                v = (1-gamma)*v + gamma*(r_np1 - r_n)**2

                gk = np.sqrt(2) / (k*k*np.sqrt(v/2 + eps))
                gk = np.clip(gk, 0.0, 1.0)

                betahat_new = (1 - gk)*betahat + gk*r_np1

                # variance of betahat updates
                v2 = (1 - gk)*v2 + gk*(betahat_new - betahat)**2

                betahat = betahat_new

        # ---------- sparse sampling ----------
        if betahat > 0:
            gap = int(C_GAP / betahat)
            if gap > 0 and step - last_sample_n >= gap:
                sample_n.append(step)
                sample_L.append(L)
                last_sample_n = step

# ============================================================
# POST-PROCESSING
# ============================================================

loss_hist = np.array(loss_hist)
beta_hist = np.array(beta_hist)
sample_n = np.array(sample_n)
sample_L = np.array(sample_L)

if len(sample_n) > 6:
    sample_n = sample_n[5:]
    sample_L = sample_L[5:]

# final beta
beta_final = np.mean(beta_hist[-200:])

# ============================================================
# LEAST SQUARES ESTIMATION OF α, ℓ
# ============================================================

x_ls = np.exp(-beta_final * sample_n)
X = np.vstack([x_ls, np.ones_like(x_ls)]).T
theta, _, _, _ = np.linalg.lstsq(X, sample_L, rcond=None)
alpha_hat, l_hat = theta

residuals = sample_L - X @ theta
sigma2_hat = np.sum(residuals**2) / (len(sample_L) - 2)
cov = sigma2_hat * np.linalg.inv(X.T @ X)
std_l_OLS = np.sqrt(cov[1,1])

# ============================================================
# PROPAGATE β UNCERTAINTY INTO ℓ
# ============================================================

var_beta = v2 / 2.0
dell_dbeta = alpha_hat * sample_n * np.exp(-betahat * sample_n)
var_l_beta = np.sum(dell_dbeta**2) * var_beta / len(sample_n)**2

# total standard deviation including beta uncertainty
std_l_total = np.sqrt(std_l_OLS**2 + var_l_beta)

# ============================================================
# OUTPUT
# ============================================================

print(f"beta_hat  = {beta_final:.6f}")
print(f"alpha_hat = {alpha_hat:.6f}")
print(f"l_hat     = {l_hat:.6f}")
print(f"std(l)    = {std_l_total:.6f}")

# ============================================================
# PLOTS
# ============================================================

plt.figure(figsize=(12,9))

plt.subplot(3,1,1)
plt.plot(loss_hist, alpha=0.7, label="loss")
plt.scatter(sample_n, sample_L, s=15, color="red", label="LS samples")
plt.title("Training loss (highly irregular)")
plt.ylabel("Loss")
plt.legend()

plt.subplot(3,1,2)
plt.plot(beta_hist, label=r"$\hat\beta_n$")
plt.axhline(beta_final, color="k", linestyle="--", label="final β")
plt.ylabel("beta")
plt.legend()

plt.subplot(3,1,3)
plt.plot(loss_hist, alpha=0.3, label="loss")
plt.axhline(l_hat, color="red", label=r"$\hat\ell$")
plt.fill_between(
    np.arange(len(loss_hist)),
    l_hat - std_l_total,
    l_hat + std_l_total,
    color="red",
    alpha=0.2,
    label="±1σ (total)"
)
plt.ylabel("Loss")
plt.xlabel("Step")
plt.legend()

plt.tight_layout()
plt.show()
