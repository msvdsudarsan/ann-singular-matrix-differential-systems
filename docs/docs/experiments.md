# Numerical Experiments

This document summarizes the numerical experiments implemented
in this repository using Physics-Informed Neural Networks (PINNs).

The experiments are organized to cover three important classes
of singular matrix differential systems.

---

## 1. Singularly Perturbed System

A boundary value problem with a small perturbation parameter
is considered. Such problems exhibit boundary layer behavior
near the domain endpoints.

The PINN model is trained using interior collocation points
and boundary conditions. The experiment demonstrates how
the neural network captures sharp solution transitions
without requiring mesh refinement.

---

## 2. Pantograph-Type Delay Differential Equation

A delay differential equation with proportional delay is studied.
The solution at a scaled time argument is directly handled
by the neural network.

This experiment shows that PINNs can avoid interpolation-based
numerical errors typically encountered in classical solvers.

---

## 3. Matrix Riccati Differential Equation

A matrix Riccati differential equation arising from optimal
control is solved using a PINN formulation.

Special attention is given to preserving symmetry and numerical
stability. The results confirm that the neural network solution
maintains the essential structural properties of the matrix.

---

All experiments are implemented in MATLAB and can be executed
independently or together using the provided scripts.
