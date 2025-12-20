# An Adaptive Physics-Informed Neural Network Framework for Singular Matrix Differential Systems with Application to Controllability Analysis

This repository provides the **MATLAB implementation** accompanying the paper:  
**"An Adaptive Physics-Informed Neural Network Framework for Singular Matrix Differential Systems with Application to Controllability Analysis"**

---

## 📌 Overview

This repository implements **Physics-Informed Neural Networks (PINNs)** for numerical solution of singular and matrix differential systems.  
The focus is on solving differential equations directly using neural networks without requiring external labeled data.

---

## 📌 Problem Classes Covered

The code includes PINN solvers for:

1. **Singularly perturbed boundary value problems**
2. **Pantograph delay differential equations**
3. **Matrix Riccati differential equations arising in control**

---

## 📌 Methodological Summary

* Neural networks approximate solution functions
* Differential equation residuals are enforced via automatic differentiation
* Collocation points enforce physics constraints
* Boundary and initial conditions are embedded analytically or with loss penalties
* No external training dataset is required

---

## 📌 Repository Structure

