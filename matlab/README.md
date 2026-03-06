# PINN Framework for Singular and Matrix Differential Systems (MATLAB)

This repository provides MATLAB implementations of a Physics-Informed Neural Network (PINN) framework for solving challenging differential systems, including singularly perturbed boundary value problems, pantograph delay differential equations, and matrix Riccati differential equations arising in control theory.

The codes in this repository reproduce the numerical experiments and tables reported in the associated manuscript:

**“An Adaptive Physics-Informed Neural Network Framework for Singular Matrix Differential Systems with Application to Controllability Analysis”**

---

## 📌 Overview

Physics-Informed Neural Networks (PINNs) approximate solutions of differential equations by embedding governing equations, boundary or initial conditions, and structural constraints directly into the training loss, without requiring external data.

**Key characteristics of this framework**
- No external training data required  
- Automatic differentiation for exact derivatives  
- Mesh-free solution representation  
- Hard enforcement of boundary and initial conditions  
- Structure preservation for matrix Riccati equations  

---

## 📌 Problems Included

### 🔹 Problem 1: Singularly Perturbed Boundary Value Problem

**Equation**
\[
\epsilon y''(t) + y'(t) = 0, \quad t \in [0,1], \quad y(0)=0,\ y(1)=1
\]

**Features**
- Strong boundary layer near \( t = 0 \)  
- Hard enforcement of boundary conditions  
- Boundary-layer–aware collocation  
- Automatic differentiation for first and second derivatives  

**MATLAB file**


pinn_singular_perturbation.m


---


---

### 🔹 Problem 2: Pantograph Delay Differential Equation

**Equation**
\[
y'(t) = a y(t) + b y(\alpha t), \quad y(0)=1
\]

**Features**
- Proportional delay handled directly by the network  
- No interpolation of delayed terms  
- Hard enforcement of the initial condition  
- Reference solution generated via high-resolution RK4  

**MATLAB file**


pinn_pantograph_delay.m


---

### 🔹 Problem 3: Matrix Riccati Differential Equation

**Equation**
\[
P'(t) = -P(t)A - A^T P(t) + P(t)BR^{-1}B^T P(t) - Q,
\quad P(T) = S
\]

**Features**
- Arises from Linear Quadratic Regulator (LQR) control  
- Matrix-valued PINN output  
- Hard enforcement of terminal condition \( P(T) = S \)  
- Structure-aware formulation suitable for control applications  

**MATLAB file**

pinn_matrix_riccati.m




---

## 📌 Utility Functions

Common helper routines for collocation generation and error evaluation are provided in:

pinn_utils.m



---

## ▶️ How to Run All Experiments

To reproduce all numerical experiments reported in the paper, run the following command in MATLAB or MATLAB Online:
```matlab
run_all_experiments
