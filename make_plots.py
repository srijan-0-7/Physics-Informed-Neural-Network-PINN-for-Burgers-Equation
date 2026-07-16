"""Generate validation figures comparing PINN prediction against the
reference (ground-truth) solution of Burgers' equation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("prediction_data.npz")
x = d["x"].flatten()
t = d["t"].flatten()
Exact = d["Exact"]          # (100, 256)  rows=t, cols=x
U_pred = d["U_pred"]        # (100, 256)
X_u_train = d["X_u_train"]  # (100, 2)
err = float(d["error_l2"])

fig = plt.figure(figsize=(14, 10))

# --- Top: predicted solution heatmap with training point locations ---
ax1 = plt.subplot(2, 2, (1, 2))
im = ax1.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                 extent=[t.min(), t.max(), x.min(), x.max()],
                 origin='lower', aspect='auto')
plt.colorbar(im, ax=ax1, label='u(x,t)')
ax1.scatter(X_u_train[:, 1], X_u_train[:, 0], marker='x', c='black', s=12,
            label=f'{X_u_train.shape[0]} IC/BC training points')
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.set_title(f'PINN predicted solution u(x,t)  |  relative L2 error = {err:.2e}')
ax1.legend(loc='upper right')

# --- Bottom: solution slices at fixed t, predicted vs exact ---
slice_ts = [0.25, 0.50, 0.75]
for i, st in enumerate(slice_ts):
    ax = plt.subplot(2, 3, 4 + i)
    idx = np.argmin(np.abs(t - st))
    ax.plot(x, Exact[idx, :], 'b-', linewidth=2, label='Reference (Chebfun)')
    ax.plot(x, U_pred[idx, :], 'r--', linewidth=2, label='PINN prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title(f't = {t[idx]:.2f}')
    ax.legend()
    ax.set_ylim([-1.2, 1.2])

plt.tight_layout()
plt.savefig("pinn_burgers_validation.png", dpi=150)
print("Saved pinn_burgers_validation.png")

# --- Error heatmap ---
fig2, ax = plt.subplots(figsize=(8, 5))
abs_err = np.abs(Exact - U_pred)
im2 = ax.imshow(abs_err.T, interpolation='nearest', cmap='hot',
                 extent=[t.min(), t.max(), x.min(), x.max()],
                 origin='lower', aspect='auto')
plt.colorbar(im2, ax=ax, label='|u_exact - u_pred|')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_title('Pointwise absolute error')
plt.tight_layout()
plt.savefig("pinn_burgers_error_map.png", dpi=150)
print("Saved pinn_burgers_error_map.png")

print(f"\nFinal relative L2 error vs reference solution: {err:.6e}")
