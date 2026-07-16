"""
Independent correctness check: verifies that the autograd-computed PDE
derivatives (u_t, u_x, u_xx) used in the PINN residual match central
finite-difference approximations. This is a sanity check on the residual
*computation itself*, separate from whether training converged.
"""
import torch
import torch.nn as nn
import numpy as np

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)  # float64 needed for a stable
                                          # second-derivative finite-diff check


class PINN(nn.Module):
    def __init__(self, layers, lb, ub):
        super().__init__()
        self.lb = torch.tensor(lb, dtype=torch.float64)
        self.ub = torch.tensor(ub, dtype=torch.float64)
        modules = []
        for i in range(len(layers) - 2):
            lin = nn.Linear(layers[i], layers[i + 1])
            modules += [lin, nn.Tanh()]
        modules.append(nn.Linear(layers[-2], layers[-1]))
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
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        return u_t + u * u_x - self.nu * u_xx, u_t, u_x, u_xx


if __name__ == "__main__":
    lb = np.array([-1.0, 0.0])
    ub = np.array([1.0, 1.0])
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model = PINN(layers, lb, ub)

    sd = torch.load("pinn_burgers_model_refined.pt", map_location="cpu")
    model.load_state_dict({k: v.double() for k, v in sd.items()})
    model.eval()

    x0 = torch.tensor([[0.3], [-0.5], [0.1], [-0.2], [-0.9]])
    t0 = torch.tensor([[0.4], [0.7], [0.2], [0.5], [0.05]])

    _, ut_auto, ux_auto, uxx_auto = model.pde_residual(x0, t0)

    eps = 1e-3
    with torch.no_grad():
        u_xp, u_xm = model(x0 + eps, t0), model(x0 - eps, t0)
        u_tp, u_tm = model(x0, t0 + eps), model(x0, t0 - eps)
        u_0 = model(x0, t0)
        ux_fd = (u_xp - u_xm) / (2 * eps)
        ut_fd = (u_tp - u_tm) / (2 * eps)
        uxx_fd = (u_xp - 2 * u_0 + u_xm) / (eps ** 2)

    def rel_err(a, b):
        a, b = a.detach().flatten(), b.flatten()
        return ((a - b).abs() / (a.abs() + 1e-3)).max().item()

    err_x = rel_err(ux_auto, ux_fd)
    err_t = rel_err(ut_auto, ut_fd)
    err_xx = rel_err(uxx_auto, uxx_fd)

    print(f"max relative |u_x  (autograd) - u_x  (finite-diff)| = {err_x:.3e}")
    print(f"max relative |u_t  (autograd) - u_t  (finite-diff)| = {err_t:.3e}")
    print(f"max relative |u_xx (autograd) - u_xx (finite-diff)| = {err_xx:.3e}")

    assert err_x < 0.02 and err_t < 0.02 and err_xx < 0.05, "Gradient mismatch detected!"
    print("\nAll derivative checks PASSED (relative error < 2-5%). PDE residual computation is verified correct.")
    print("(Test points deliberately avoid x=0 near t>0.3, where the true solution develops")
    print(" a near-discontinuous shock and even the *exact* derivative is enormous/ill-conditioned")
    print(" for finite-difference comparison — that is a property of Burgers' equation, not a bug.)")
