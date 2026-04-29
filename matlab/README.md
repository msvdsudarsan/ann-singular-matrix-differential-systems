# PINN Framework for Singular Matrix Differential Systems (MATLAB)

This directory contains the MATLAB implementation of the adaptive Physics-Informed Neural Network (PINN) framework accompanying the manuscript:

**"Adaptive Physics-Informed Neural Networks for Singular Matrix Differential Systems with Algebraic Structure Preservation: Applications to Optimal Control Synthesis"**  
Sri Venkata Durga Sudarsan Madhyannapu, Pradheep Kumar S.  
*Engineering Applications of Artificial Intelligence* (Elsevier, ISSN 0952-1976) — under review, 2026.  
SSRN preprint: [doi:10.2139/ssrn.6277631](https://doi.org/10.2139/ssrn.6277631)

---

## Overview

Physics-Informed Neural Networks (PINNs) approximate solutions of differential equations by embedding the governing equations, boundary/initial conditions, and structural constraints directly into the training loss — no external data required.

**Key features of this framework:**
- Residual-adaptive collocation that concentrates points automatically near boundary layers (Algorithm 1)
- Cholesky-type parameterisation `P = L*L'` guaranteeing exact symmetry and positive semi-definiteness for all network parameters (Theorem 2)
- Hybrid PINN + `ode45` refinement recovering `1e-5` accuracy with machine-precision structural guarantees (Algorithm 2)
- Proportional-delay evaluation by direct network forward pass — no interpolation
- Automated Bayesian hyperparameter tuning

---

## Problems Included

### Problem 1: Singularly Perturbed Boundary Value Problem (Table 1, Fig. 1)

**Equation:**  
`eps*y''(t) + y'(t) = 0,   t in [0,1],   y(0)=0, y(1)=1,   eps=0.01`

**Exact solution:** `y(t) = (1 - exp(-t/eps)) / (1 - exp(-1/eps))`

**Key results:**  
- PINN MAE = (3.11 ± 2.53) × 10⁻⁶  
- ~65% of collocation points automatically concentrated in boundary layer  
- 265 adaptive collocation points total

**MATLAB file:** `pinn_singular_perturbation.m`

---

### Problem 2: Pantograph Delay Differential Equation (Table 2, Fig. 2)

**Equation:**  
`y'(t) = -y(t) + 0.5*y(0.5*t),   y(0)=1,   t in [0, 5]`

**Reference:** High-resolution RK4 with N = 6000 steps.

**Key results:**  
- PINN MAE = (9.27 ± 5.91) × 10⁻⁴  
- More than one order of magnitude improvement over MATLAB `dde23`  
- `y(0.5*t)` evaluated by direct network forward pass — no interpolation

**MATLAB files:** `pinn_pantograph_delay.m`, `pantograph_dde23.m` (dde23 baseline)

---

### Problem 3: Matrix Riccati Differential Equation (Table 3, Fig. 3)

**Equation:**  
`dP/dt = -P*A - A'*P + P*B*R^{-1}*B'*P - Q,   P(5) = I`

**System:** `A = [0 1; -1 -0.5]`, `B = [0;1]`, `Q = I`, `R = 1`

**Cholesky parameterisation:** `P_theta(t) = L_theta(t) * L_theta(t)'`  
Guarantees symmetry and positive semi-definiteness for ALL parameter values (Theorem 2).

**Key results:**  
- Standalone PINN MAE = (8.34 ± 0.73) × 10⁻²  
- Hybrid PINN + ode45 MAE = (2.17 ± 0.31) × 10⁻⁵  
- Symmetry error < 10⁻¹⁵ (identically zero by algebraic construction)  
- 150 collocation points, 4000 training epochs

**MATLAB file:** `pinn_matrix_riccati.m`

---

## Training Schedule (Section 3.6 of paper)

All problems use the same two-phase Adam schedule:
- **Phase 1:** 2000 epochs, learning rate η = 10⁻³  
- **Phase 2:** 2000 epochs, learning rate η = 10⁻⁴ (fine-tuning)  
- **Total:** 4000 epochs

---

## How to Run All Experiments

Execute the following in MATLAB R2023b (or later):

```matlab
run_all_experiments
```

This sequentially runs all three problems and prints results alongside the paper's reported values.

**Requirements:** MATLAB R2023b, Deep Learning Toolbox, Optimization Toolbox.

---

## File List

| File | Purpose |
|---|---|
| `pinn_singular_perturbation.m` | Adaptive PINN for singularly perturbed BVP (Table 1) |
| `pinn_pantograph_delay.m` | PINN for pantograph DDE with proportional delay (Table 2) |
| `pinn_matrix_riccati.m` | Cholesky-PINN + hybrid refinement for Riccati (Table 3) |
| `pantograph_dde23.m` | MATLAB `dde23` baseline (Table 2 comparison) |
| `compare_finite_difference.m` | Finite difference baseline (Table 1 comparison) |
| `bvp4c_singular_test.m` | `bvp4c` baseline (Table 1 comparison) |
| `compare_uniform_pinn.m` | Uniform-collocation PINN baseline (Table 1 comparison) |
| `compare_rk4_pantograph.m` | RK4 + interpolation baseline (Table 2 comparison) |
| `loss_evolution_comparison.m` | Training loss curves (Fig. 4) |
| `pinn_utils.m` | Shared utility functions |
| `run_all_experiments.m` | Master script — runs all experiments |
