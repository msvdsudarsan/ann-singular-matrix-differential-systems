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
├── pinn_singular_perturbation.m   % Singularly perturbed BVP PINN
├── pinn_pantograph_delay.m        % Pantograph delay PINN
├── pinn_matrix_riccati.m          % Matrix Riccati PINN
├── pinn_utils.m                   % Utility functions
├── run_all_experiments.m          % Runs all experiments
├── results/                       % Generated numerical outputs
└── figures/                       % Figures used in the manuscript

---

## Numerical Results (Summary)

The numerical results reported in the JAMC manuscript were generated using the
MATLAB scripts provided in this repository with fixed random seeds.

Key representative results include:

- **Singularly Perturbed BVP (ε = 0.01)**  
  Adaptive PINN achieves **MAE ≈ 2.98 × 10⁻⁵**.

- **Pantograph Delay Differential Equation**  
  PINN achieves **MAE ≈ 1.96 × 10⁻²**.

- **Matrix Riccati Differential Equation**  
  PINN achieves **MAE ≈ 9.97 × 10⁻²**, while preserving symmetry and positive
  definiteness throughout training.

