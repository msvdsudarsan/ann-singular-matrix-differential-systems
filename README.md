# An Adaptive Physics-Informed Neural Network Framework for Singular Matrix Differential Systems with Applications to Controllability Analysis

This repository provides the MATLAB implementation accompanying the paper:

**“An Adaptive Physics-Informed Neural Network Framework for Singular Matrix Differential Systems with Applications to Controllability Analysis”**

The code implements Physics-Informed Neural Networks (PINNs) for solving singular and matrix differential systems arising in control theory and applied mathematics.

---

## 📌 Overview

This repository focuses on solving differential equations directly using Physics-Informed Neural Networks (PINNs) without relying on external labeled data. The governing equations, boundary or terminal conditions, and structural constraints are embedded into the training process through automatic differentiation.

The implementation serves as a computational companion to the manuscript submitted to the  
**Journal of Applied Mathematics and Computing (JAMC)**.

---

## 📌 Problem Classes Covered

The repository includes PINN solvers for the following three classes of problems:

### 1. Singularly Perturbed Boundary Value Problems
- Problems exhibiting boundary layers due to small perturbation parameters  
- Boundary-layer–aware collocation improves accuracy near sharp transitions  

### 2. Pantograph Delay Differential Equations
- Differential equations with proportional delay terms of the form \( y(\alpha t) \)  
- PINNs avoid interpolation errors inherent in classical time-stepping solvers  

### 3. Matrix Riccati Differential Equations
- Arising in optimal control and Linear Quadratic Regulator (LQR) design  
- A structure-preserving formulation ensures symmetry and positive definiteness  

---

## 📌 Methodological Summary

- Neural networks approximate the solution functions directly  
- Governing differential equations are enforced via residual minimization  
- Derivatives are computed using automatic differentiation  
- Boundary and terminal conditions are imposed analytically via hard constraints  
- No external training datasets are required  
- Adaptive collocation refines points automatically in regions of rapid variation  
- Matrix Riccati equations are handled using a structure-aware formulation  

---

## 📌 Repository Structure

```text
matlab/
├── pinn_singular_perturbation.m % Adaptive PINN for singularly perturbed BVP
├── pinn_pantograph_delay.m % PINN for pantograph delay equation
├── pinn_matrix_riccati.m % Structure-preserving Riccati PINN
├── pinn_utils.m % Shared utility functions

├── compare_finite_difference.m % Finite Difference baseline (BVP)
├── compare_uniform_pinn.m % Uniform PINN baseline (no adaptivity)
├── compare_rk4_pantograph.m % RK4 + interpolation baseline
├── bvp4c_singular_test.m % MATLAB bvp4c reference solver (Gap-1)

├── run_all_experiments.m % Runs all PINN experiments
├── results/ % Generated numerical outputs
└── figures/ % Figures used in the manuscript
---

## Numerical Results (Summary)

The numerical results reported in the JAMC manuscript were generated using the
MATLAB scripts provided in this repository with fixed random seeds.
For PINN-based methods, results are reported as mean ± standard deviation over
three independent runs.

### Singularly Perturbed Boundary Value Problem (ε = 0.01)
- Finite Difference (uniform):  
  MAE = 8.12 × 10⁻⁴, Max Error = 3.46 × 10⁻²
- bvp4c (adaptive MATLAB solver):  
  MAE = 1.85 × 10⁻⁶, Max Error = 2.41 × 10⁻⁴
- Adaptive PINN (this work):  
  MAE = (3.11 ± 2.53) × 10⁻⁶,  
  Max Error = (2.36 ± 2.09) × 10⁻⁵  

Notably, while `bvp4c` achieves slightly lower mean error, the adaptive PINN
attains an order-of-magnitude smaller maximum error, indicating superior
resolution of sharp boundary-layer features.

### Pantograph Delay Differential Equation
- RK4 + interpolation baseline:  
  MAE = 1.04 × 10⁻¹, Max Error = 2.64 × 10⁻¹
- PINN (this work):  
  MAE = (2.18 ± 0.07) × 10⁻²,  
  Max Error = (1.29 ± 0.05) × 10⁻¹

### Matrix Riccati Differential Equation
- Structure-preserving PINN (this work):  
  MAE = (9.84 ± 1.15) × 10⁻²  

The proposed Riccati PINN guarantees symmetry and positive definiteness of the
solution throughout training, at the cost of reduced numerical accuracy compared
to classical solvers such as `ode45`.
