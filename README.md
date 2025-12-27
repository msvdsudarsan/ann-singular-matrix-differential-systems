# An Adaptive Physics-Informed Neural Network Framework for Singular Matrix Differential Systems with Application to Controllability Analysis

This repository provides the MATLAB implementation accompanying the paper:

**“An Adaptive Physics-Informed Neural Network Framework for Singular Matrix Differential Systems with Application to Controllability Analysis”**

The code implements Physics-Informed Neural Networks (PINNs) for solving singular and matrix differential systems relevant to control theory and applied mathematics.

---

## 📌 Overview

This repository focuses on solving differential equations directly using physics-informed neural networks without relying on external labeled data.  
The governing equations, boundary conditions, and structural constraints are embedded into the training process through automatic differentiation.

The implementation is intended as a **computational companion** to the manuscript submitted to the *Journal of Computational and Applied Mathematics (JCAM)*.

---

## 📌 Problem Classes Covered

The repository includes PINN solvers for the following three classes of problems:

1. **Singularly Perturbed Boundary Value Problems**
   - Problems exhibiting boundary layers due to small perturbation parameters.
   - Adaptive collocation improves accuracy near sharp transitions.

2. **Pantograph Delay Differential Equations**
   - Differential equations with proportional delay terms of the form \( y(\alpha t) \).
   - PINNs avoid interpolation errors inherent in classical solvers.

3. **Matrix Riccati Differential Equations**
   - Arising in optimal control and LQR design.
   - A structure-preserving parameterization ensures symmetry and positive definiteness.

---

## 📌 Methodological Summary

- Neural networks approximate the solution functions directly.
- Governing differential equations are enforced via residual minimization.
- Derivatives are computed using automatic differentiation.
- Boundary and initial conditions are imposed analytically or via loss penalties.
- No external training datasets are required.
- Adaptive collocation refines points automatically in regions of rapid variation.
- Matrix Riccati equations are solved using a structure-preserving formulation.

---

## 📌 Repository Structure

