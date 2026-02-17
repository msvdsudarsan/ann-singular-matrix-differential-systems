# Adaptive Physics-Informed Neural Networks for Singular Matrix Differential Systems with Applications to Optimal Control Synthesis

[![Paper](https://img.shields.io/badge/Journal-Advances%20in%20Engineering%20Software-blue)](https://www.sciencedirect.com/journal/advances-in-engineering-software)
[![SSRN](https://img.shields.io/badge/Preprint-SSRN-orange)](https://ssrn.com)
[![MATLAB](https://img.shields.io/badge/Code-MATLAB-red)](https://www.mathworks.com)

## Authors

- **Sri Venkata Durga Sudarsan Madhyannapu** — NRI Institute of Technology & JNTUK, Andhra Pradesh, India
- **Pradheep Kumar S.** — SRM University AP, Andhra Pradesh, India

---

## Abstract

This repository contains MATLAB source code for the paper submitted to *Advances in Engineering Software* (Elsevier), Manuscript ID: ADES-D-26-00359.

We develop an adaptive physics-informed neural network (PINN) framework for three representative classes of singular matrix differential systems:

1. **Singularly Perturbed Boundary Value Problems** — boundary layer resolution via adaptive collocation
2. **Pantograph Delay Differential Equations** — non-local coupling handled via continuous neural representation
3. **Matrix Riccati Differential Equations** — structure-preserving Cholesky-type parameterization for LQR control synthesis

---

## Repository Structure

```
ann-singular-matrix-differential-systems/
│
├── README.md                          ← This file
│
├── singular_bvp/
│   ├── pinn_singular_bvp.m            ← PINN solver for singularly perturbed BVP
│   └── adaptive_collocation.m         ← Adaptive refinement algorithm
│
├── pantograph_dde/
│   ├── pinn_pantograph.m              ← PINN solver for pantograph DDE
│   └── pantograph_dde23.m             ← MATLAB dde23 comparison solver
│
├── riccati/
│   ├── pinn_riccati_structure.m       ← Structure-preserving PINN (Cholesky)
│   └── hybrid_riccati.m              ← Hybrid PINN + ode45 refinement
│
├── aerospace/
│   └── spacecraft_trajectory_6d.m    ← 6D spacecraft trajectory optimization
│
└── figures/
    ├── fig_singular_bvp_comparison.pdf
    ├── fig_pantograph_comparison.pdf
    ├── figure_riccati_trace.pdf
    ├── fig_loss_evolution_comparison.pdf
    └── fig_aerospace_trajectory.pdf
```

---

## Key Results

| Problem | Method | MAE |
|---------|--------|-----|
| Singularly Perturbed BVP | PINN (adaptive) | 3.11 × 10⁻⁶ |
| Singularly Perturbed BVP | bvp4c | 1.85 × 10⁻⁶ |
| Pantograph DDE | PINN | 9.27 × 10⁻⁴ |
| Pantograph DDE | dde23 (MATLAB) | 1.10 × 10⁻² |
| Pantograph DDE | RK4 + interpolation | 1.04 × 10⁻¹ |
| Matrix Riccati | PINN + Hybrid | 2.17 × 10⁻⁵ |
| Matrix Riccati | Symmetry error | < 10⁻¹⁵ |

---

## Requirements

- MATLAB R2021b or later
- Deep Learning Toolbox
- Optimization Toolbox (for Bayesian hyperparameter tuning)

---

## Usage

```matlab
% Example: Run PINN for pantograph DDE
cd pantograph_dde/
run pinn_pantograph.m

% Example: dde23 comparison
run pantograph_dde23.m
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{madhyannapu2026adaptive,
  title={Adaptive Physics-Informed Neural Networks for Singular Matrix 
         Differential Systems with Applications to Optimal Control Synthesis},
  author={Madhyannapu, Sri Venkata Durga Sudarsan and Pradheep Kumar, S.},
  journal={Advances in Engineering Software},
  year={2026},
  note={Under Review, Manuscript ID: ADES-D-26-00359}
}
```

---

## Related Publication

S.V.D.S. Madhyannapu, V.S. Putcha, G.V.S.R. Deekshitulu,
"Equivalence of Kalman and Hewer controllability of Lyapunov matrix periodic systems,"
*i-Manager's Journal of Mathematics*, 14(1), 2025.

---

## License

This code is provided for academic research purposes only.
