"""Resume training the saved PINN with additional L-BFGS iterations to
further reduce the residual loss and validation error."""
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import time
import json

torch.manual_seed(1234)
np.random.seed(1234)
torch.set_num_threads(1)
device = torch.device("cpu")

data = sio.loadmat("data/burgers_shock.mat")
t_ref = data["t"].flatten()[:, None]
x_ref = data["x"].flatten()[:, None]
Exact = np.real(data["usol"]).T

X, T = np.meshgrid(x_ref, t_ref)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]
lb = X_star.min(0)
ub = X_star.max(0)

N_u = 100
N_f = 6000  # increase collocation density for the refinement pass

xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
uu1 = -np.sin(np.pi * xx1[:, 0:1])
xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
uu2 = np.zeros((xx2.shape[0], 1))
xx3 = np.hstack((X[:, -1:], T[:, -1:]))
uu3 = np.zeros((xx3.shape[0], 1))
X_u_train_all = np.vstack([xx1, xx2, xx3])
u_train_all = np.vstack([uu1, uu2, uu3])
idx = np.random.choice(X_u_train_all.shape[0], N_u, replace=False)
X_u_train = X_u_train_all[idx, :]
u_train = u_train_all[idx, :]


def latin_hypercube_sample(n_dims, n_samples):
    result = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = np.random.permutation(n_samples)
        u = np.random.rand(n_samples)
        result[:, d] = (perm + u) / n_samples
    return result


X_f_train = lb + (ub - lb) * latin_hypercube_sample(2, N_f)
X_f_train = np.vstack((X_f_train, X_u_train))


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
        self.nu = 0.01 / np.pi

    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
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


layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
model = PINN(layers, lb, ub).to(device)
model.load_state_dict(torch.load("pinn_burgers_model.pt", map_location=device))
print("Loaded checkpoint.")

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


optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=1.0, max_iter=3000, max_eval=3000,
    history_size=100, tolerance_grad=1e-10, tolerance_change=1e-13,
    line_search_fn="strong_wolfe"
)
it_count = [0]
history = []


def closure():
    optimizer_lbfgs.zero_grad()
    loss_u, loss_f = compute_loss()
    loss = loss_u + loss_f
    loss.backward()
    it_count[0] += 1
    if it_count[0] % 200 == 0:
        print(f"refine LBFGS it {it_count[0]:5d} | loss_u {loss_u.item():.4e} | loss_f {loss_f.item():.4e} | total {loss.item():.4e}")
        history.append(loss.item())
    return loss


t0 = time.time()
optimizer_lbfgs.step(closure)
print(f"Refinement took {time.time()-t0:.1f}s, iters={it_count[0]}")

model.eval()
with torch.no_grad():
    u_pred_star = model(x_star_t, t_star_t).cpu().numpy()

error_l2 = np.linalg.norm(u_star - u_pred_star, 2) / np.linalg.norm(u_star, 2)
print(f"Refined relative L2 error: {error_l2:.6e}")

results = {
    "relative_l2_error": float(error_l2),
    "n_u": N_u,
    "n_f": N_f,
    "refine_lbfgs_iters": it_count[0],
}
with open("results_refined.json", "w") as f:
    json.dump(results, f, indent=2)

np.savez("prediction_data.npz",
         x=x_ref, t=t_ref, Exact=Exact,
         U_pred=u_pred_star.reshape(X.shape),
         X_u_train=X_u_train, error_l2=error_l2)

torch.save(model.state_dict(), "pinn_burgers_model_refined.pt")
print("Saved refined results.")
