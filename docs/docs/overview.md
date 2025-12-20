# Overview: Physics-Informed Neural Networks for Singular Matrix Systems

This document summarizes the framework and methodology implemented in the repository.

---

## What is a PINN?

Physics-Informed Neural Networks (PINNs) approximate solutions of differential equations
by minimizing a loss based on the equation residuals instead of data mismatch.

---

## Problem Classes

1. **Singularly Perturbed ODE/BVPs**  
   Stiff boundary-layer problems with small parameters.

2. **Delayed Differential Equations**  
   Including pantograph type delays (e.g., y'(t) = y(t) + y(t/2)).

3. **Matrix Differential Equations**  
   Like matrix Riccati equations from control theory.

---

## Key features

* Residual evaluation using automatic differentiation
* Boundary/initial condition incorporation
* Mesh-free solution formulation
* Simple collocation strategies
