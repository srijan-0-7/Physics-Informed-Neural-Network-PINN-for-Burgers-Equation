# Physics-Informed-Neural-Network-PINN-for-Burgers-Equation
## Overview
This project provides a PyTorch implementation of a Physics-Informed Neural Network (PINN) designed to solve the 1D viscous Burgers' equation. It reproduces the canonical example from the foundational paper by Raissi et al. (2019). The primary objective is to train a neural network to satisfy the governing Partial Differential Equation (PDE) across a continuous domain, relying solely on boundary and initial conditions without being exposed to any interior solution data during training.

Burgers' equation serves as an excellent benchmark because it develops a steep, near-discontinuous shock front over time, providing a rigorous stress test for the network's ability to capture complex, nonlinear physical phenomena.

**Reference:**
> Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.

## Mathematical Formulation
The network $u_{\theta}(x, t)$ is trained to satisfy the following PDE:

$$ u_t + u \cdot u_x - \frac{0.01}{\pi} \cdot u_{xx} = 0 \quad \text{for } x \in [-1, 1], \, t \in [0, 1] $$

Subject to the Initial Condition (IC) and Dirichlet Boundary Conditions (BC):
* **IC:** $u(x, 0) = -\sin(\pi x)$
* **BC:** $u(-1, t) = u(1, t) = 0$

## Methodology & Architecture
* **Architecture:** Fully connected neural network with 2 input features ($x$ and $t$), 8 hidden layers comprising 20 neurons each, and a single output predicting $u$. It utilizes `tanh` activation functions and Xavier-normal weight initialization.
* **Loss Function:** Minimizes a composite loss:
    1.  **Data Loss:** Mean Squared Error (MSE) evaluated at 100 randomly sampled points from the initial and boundary conditions.
    2.  **Physics Loss:** MSE of the PDE residual evaluated at 4,000 (later refined to 6,000) internal collocation points generated via Latin Hypercube Sampling. The residual is computed using PyTorch's autograd engine to find the exact analytical derivatives ($u_t, u_x, u_{xx}$).
* **Optimization:** A two-stage strategy: an initial phase using the Adam optimizer for robust initial descent (3,000 iterations), followed by fine-tuning with the L-BFGS optimizer for high-precision convergence.

## Repository Structure

| File | Description |
|---|---|
| `pinn_burgers.py` | Main training script encompassing data loading, network definition, Adam and L-BFGS optimization stages, and validation. |
| `refine_train.py` | Script to resume training from a checkpoint with a higher density of collocation points (6,000) and additional L-BFGS iterations for refinement. |
| `make_plots.py` | Utility script to generate visualization figures comparing the PINN predictions against the exact reference solution. |
| `verify_gradients.py` | An independent script utilizing finite-difference approximations to mathematically verify that the autograd-computed derivatives are accurate. |
| `data/burgers_shock.mat` | The exact reference solution computed independently using a spectral method (Chebfun), used strictly for validation and never seen during training. |
| `pinn_burgers_model_refined.pt` | The final saved model weights after the refinement training stage. |

## Outputs
Upon successful execution of the training and plotting scripts, the following outputs are generated:
* `results_refined.json`: Contains the quantitative metrics from the final evaluation, including the total iterations and the final relative error.
* `pinn_burgers_validation.png`: A comprehensive visualization featuring a heatmap of the predicted solution across space and time, overlaid with the spatial locations of the training data points. It also includes 1D slices at specific time steps ($t = 0.25, 0.50, 0.75$) comparing the network's prediction against the exact Chebfun reference solution.
* `pinn_burgers_error_map.png`: A heatmap depicting the pointwise absolute error between the prediction and the ground truth.

## Conclusion
The project successfully demonstrates the efficacy of Physics-Informed Neural Networks in solving nonlinear PDEs without relying on simulated interior data. The final refined model achieved a relative L2 error of **3.98 × 10⁻³** against the independent reference solution across a 25,600-point grid.

The visual outputs and the pointwise error map confirm that the model accurately captures the dynamics of the Burgers' equation. The error is near machine precision in smooth regions of the domain and is primarily concentrated along the steep shock front, which aligns exactly with the expected failure modes of PINNs on this specific benchmark. Furthermore, the `verify_gradients.py` script confirms that the underlying autograd mathematics driving the physics loss are functioning correctly.
