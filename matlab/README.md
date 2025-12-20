# PINN Framework for Singular and Matrix Differential Systems

This repository provides MATLAB implementations of a **Physics-Informed Neural Network (PINN)** framework for solving challenging differential systems, including singularly perturbed problems, delay differential equations, and matrix Riccati equations arising in control theory.

The codes reproduce the numerical experiments reported in the associated manuscript.

---

## Problems Included

### Problem 1: Singularly Perturbed Boundary Value Problem
- Equation:  
  \[
  \epsilon y''(t) + y'(t) = 0,\quad t\in[0,1],\quad y(0)=0,\; y(1)=1
  \]
- Boundary layer near \( t=0 \)
- Adaptive collocation strategy
- File: `pinn_singular_perturbation.m`

---

### Problem 2: Pantograph Delay Differential Equation
- Equation:  
  \[
  y'(t) = y(t) + y(t/2),\quad y(0)=1
  \]
- Proportional delay handled directly by the network
- No interpolation required
- File: `pinn_pantograph_delay.m`

---

### Problem 3: Matrix Riccati Differential Equation
- Equation:
  \[
  X'(t) = A^T X + X A - X B R^{-1} B^T X + Q,\quad X(0)=I
  \]
- Arises from Linear Quadratic Regulator (LQR) control
- Matrix-valued PINN with hard initial condition enforcement
- File: `pinn_matrix_riccati.m`

---

## How to Run All Experiments

Run the following command in MATLAB or MATLAB Online:

```matlab
run_all_experiments
