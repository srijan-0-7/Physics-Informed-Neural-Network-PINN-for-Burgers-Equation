"""
Physics-Informed Neural Network (PINN) for the 1D viscous Burgers' equation.

Reproduces the canonical example from:
Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).
"Physics-informed neural networks: A deep learning framework for solving
forward and inverse problems involving nonlinear partial differential
equations." Journal of Computational Physics, 378, 686-707.

PDE:      u_t + u*u_x - (0.01/pi)*u_xx = 0,   x in [-1, 1], t in [0, 1]
IC:       u(x, 0) = -sin(pi*x)
BC:       u(-1, t) = u(1, t) = 0   (Dirichlet)

The network is trained ONLY on the initial/boundary data plus the PDE
residual evaluated at randomly sampled collocation points -- it never sees
the interior solution during training. We then validate against the
reference solution (data/burgers_shock.mat) computed independently by the
original authors with a spectral method (Chebfun), which is NOT used in
training at all, only for evaluation.
"""

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import time
import json

torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------------------------
# 1. Load reference data (ground truth, used ONLY for validation)
# ----------------------------------------------------------------------
data = sio.loadmat("data/burgers_shock.mat")
t_ref = data["t"].flatten()[:, None]      # (100, 1)
x_ref = data["x"].flatten()[:, None]      # (256, 1)
Exact = np.real(data["usol"]).T           # (100, 256)  -> rows=t, cols=x

X, T = np.meshgrid(x_ref, t_ref)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))   # (25600, 2)
u_star = Exact.flatten()[:, None]                                  # (25600, 1)

lb = X_star.min(0)   # domain lower bound [x_min, t_min]
ub = X_star.max(0)   # domain upper bound [x_max, t_max]

# ----------------------------------------------------------------------
# 2. Build training data: initial condition + boundary conditions
# ----------------------------------------------------------------------
N_u = 100      # number of IC/BC points
N_f = 4000     # number of collocation (PDE residual) points

# Initial condition slice: t = 0, u = -sin(pi*x)
xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))           # (256, 2)
uu1 = -np.sin(np.pi * xx1[:, 0:1])

# Boundary x = -1
xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))                # (100, 2)
uu2 = np.zeros((xx2.shape[0], 1))

# Boundary x = 1
xx3 = np.hstack((X[:, -1:], T[:, -1:]))                # (100, 2)
uu3 = np.zeros((xx3.shape[0], 1))

X_u_train_all = np.vstack([xx1, xx2, xx3])
u_train_all = np.vstack([uu1, uu2, uu3])

idx = np.random.choice(X_u_train_all.shape[0], N_u, replace=False)
X_u_train = X_u_train_all[idx, :]
u_train = u_train_all[idx, :]

def latin_hypercube_sample(n_dims, n_samples):
    """Simple, self-contained Latin Hypercube Sampling in [0, 1]^n_dims."""
    result = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = np.random.permutation(n_samples)
        u = np.random.rand(n_samples)
        result[:, d] = (perm + u) / n_samples
    return result


# Collocation points via Latin Hypercube Sampling over the domain
X_f_train = lb + (ub - lb) * latin_hypercube_sample(2, N_f)
X_f_train = np.vstack((X_f_train, X_u_train))  # also enforce residual at IC/BC pts

# ----------------------------------------------------------------------
# 3. PINN model definition
# ----------------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self, layers, lb, ub):
        super().__init__()
        self.lb = torch.tensor(lb, dtype=torch.float32, device=device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=device)

        modules = []
        for i in range(len(layers) - 2):
            lin = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            modules.append(lin)
            modules.append(nn.Tanh())
        last = nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_normal_(last.weight)
        nn.init.zeros_(last.bias)
        modules.append(last)
        self.net = nn.Sequential(*modules)

        self.nu = 0.01 / np.pi  # known viscosity coefficient

    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
        # normalize inputs to [-1, 1]
        Xn = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(Xn)

    def pde_residual(self, x, t):
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        u = self.forward(x, t)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True, retain_graph=True)[0]

        f = u_t + u * u_x - self.nu * u_xx
        return f


layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]  # 8 hidden layers x 20 neurons, as in the paper
model = PINN(layers, lb, ub).to(device)

# ----------------------------------------------------------------------
# 4. Prepare tensors
# ----------------------------------------------------------------------
x_u = torch.tensor(X_u_train[:, 0:1], dtype=torch.float32, device=device)
t_u = torch.tensor(X_u_train[:, 1:2], dtype=torch.float32, device=device)
u_d = torch.tensor(u_train, dtype=torch.float32, device=device)

x_f = torch.tensor(X_f_train[:, 0:1], dtype=torch.float32, device=device)
t_f = torch.tensor(X_f_train[:, 1:2], dtype=torch.float32, device=device)

x_star_t = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
t_star_t = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)

mse = nn.MSELoss()


def compute_loss():
    u_pred = model(x_u, t_u)
    f_pred = model.pde_residual(x_f, t_f)
    loss_u = mse(u_pred, u_d)
    loss_f = mse(f_pred, torch.zeros_like(f_pred))
    return loss_u, loss_f


# ----------------------------------------------------------------------
# 5. Stage 1: Adam optimizer
# ----------------------------------------------------------------------
print("\n=== Stage 1: Adam optimization ===")
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
n_adam = 3000
history = []
t0 = time.time()
for it in range(n_adam):
    optimizer_adam.zero_grad()
    loss_u, loss_f = compute_loss()
    loss = loss_u + loss_f
    loss.backward()
    optimizer_adam.step()
    if it % 1000 == 0 or it == n_adam - 1:
        print(f"Adam it {it:5d} | loss_u {loss_u.item():.4e} | loss_f {loss_f.item():.4e} | total {loss.item():.4e}")
    history.append(loss.item())
print(f"Adam stage took {time.time()-t0:.1f}s")

# ----------------------------------------------------------------------
# 6. Stage 2: L-BFGS fine-tuning (matches original paper's optimizer)
# ----------------------------------------------------------------------
print("\n=== Stage 2: L-BFGS optimization ===")
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=1.0, max_iter=1500, max_eval=1500,
    history_size=50, tolerance_grad=1e-9, tolerance_change=1e-12,
    line_search_fn="strong_wolfe"
)

lbfgs_iter = [0]

def closure():
    optimizer_lbfgs.zero_grad()
    loss_u, loss_f = compute_loss()
    loss = loss_u + loss_f
    loss.backward()
    lbfgs_iter[0] += 1
    if lbfgs_iter[0] % 200 == 0:
        print(f"LBFGS it {lbfgs_iter[0]:5d} | loss_u {loss_u.item():.4e} | loss_f {loss_f.item():.4e} | total {loss.item():.4e}")
        history.append(loss.item())
    return loss

t1 = time.time()
optimizer_lbfgs.step(closure)
print(f"L-BFGS stage took {time.time()-t1:.1f}s, total iters: {lbfgs_iter[0]}")

# ----------------------------------------------------------------------
# 7. Validation against reference (ground-truth) solution
# ----------------------------------------------------------------------
model.eval()
with torch.no_grad():
    u_pred_star = model(x_star_t, t_star_t).cpu().numpy()

error_l2 = np.linalg.norm(u_star - u_pred_star, 2) / np.linalg.norm(u_star, 2)
print(f"\n=== Validation ===")
print(f"Relative L2 error on full reference grid (25600 points, none used in training): {error_l2:.6e}")

# sanity bounds: the original paper reports ~6.7e-4 for this exact setup.
results = {
    "relative_l2_error": float(error_l2),
    "n_u": N_u,
    "n_f": N_f,
    "adam_iters": n_adam,
    "lbfgs_iters": lbfgs_iter[0],
    "final_loss": float(history[-1]),
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save predictions + everything needed for plotting
np.savez("prediction_data.npz",
         x=x_ref, t=t_ref, Exact=Exact,
         U_pred=u_pred_star.reshape(T.shape, order='C'),
         X_u_train=X_u_train, error_l2=error_l2,
         history=np.array(history))

torch.save(model.state_dict(), "pinn_burgers_model.pt")
print("\nSaved: results.json, prediction_data.npz, pinn_burgers_model.pt")
