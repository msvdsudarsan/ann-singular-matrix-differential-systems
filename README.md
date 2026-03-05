# Adaptive Physics-Informed Neural Networks for Singular Matrix Differential Systems with Applications to Optimal Control Synthesis

## Authors
- **Sri Venkata Durga Sudarsan Madhyannapu**
- **Pradheep Kumar S.**

## Affiliations
1. Freshmen Engineering Department, NRI Institute of Technology, Pothavarappadu, Agiripalli, Eluru District 521212, Andhra Pradesh, India
2. Research Scholar, Jawaharlal Nehru Technological University Kakinada, Andhra Pradesh, India
3. School of Basic Sciences, SRM University AP, Neerukonda, Mangalagiri, Guntur–522240, Andhra Pradesh, India

## Manuscript Information
| | |
|---|---|
| **Journal** | Neurocomputing, Elsevier, ISSN: 0925-2312 |
| **Manuscript ID** | NEUCOM-D-26-03849 |
| **Status** | Under Review, 2026 |

## Overview
Adaptive PINN framework for three problem classes:
1. **Singularly Perturbed BVPs** (ε = 0.01) — MAE 3.11×10⁻⁶
2. **Pantograph Delay Differential Equations** (α = 0.5) — MAE 9.27×10⁻⁴ vs 1.10×10⁻² for dde23
3. **Matrix Riccati Equations** (LQR control) — MAE 2.17×10⁻⁵, symmetry error < 10⁻¹⁵

## Network Architecture
- 3 hidden layers × 50 neurons (tanh activation)
- Adam optimizer: 2000 epochs (lr=1e-3) + 2000 epochs (lr=1e-4)
- Cholesky parameterization P(t)=L(t)L(t)ᵀ for structure preservation

## Repository Structure
```
ann-singular-matrix-systems/
├── README.md
├── pinn_singular_perturbation.m
├── pinn_pantograph_delay.m
├── pinn_matrix_riccati.m
├── bvp4c_singular_test.m
├── compare_finite_difference.m
├── compare_uniform_pinn.m
├── pinn_utils.m
└── run_all_experiments.m
```

## Quick Start
**Prerequisites:** MATLAB R2023b with Deep Learning Toolbox
```matlab
run_all_experiments();          % Run all three problems
pinn_singular_perturbation();   % Problem 1: BVP
pinn_pantograph_delay();        % Problem 2: DDE
pinn_matrix_riccati();          % Problem 3: Riccati/LQR
```

## Citation
```bibtex
@article{sudarsan2026pinn,
  title   = {Adaptive Physics-Informed Neural Networks for Singular Matrix
             Differential Systems with Applications to Optimal Control Synthesis},
  author  = {Madhyannapu, Sri Venkata Durga Sudarsan and Pradheep Kumar, S.},
  journal = {Neurocomputing},
  publisher = {Elsevier},
  issn    = {0925-2312},
  year    = {2026},
  note    = {Under Review, Manuscript ID: NEUCOM-D-26-03849}
}
```

## License
MIT License — provided for academic research purposes only.
