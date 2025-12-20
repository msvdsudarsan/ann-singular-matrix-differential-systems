# Overview of the PINN Framework

This document provides a concise overview of the Physics-Informed Neural Network (PINN) framework implemented in this repository for solving singular and matrix differential systems.

---

## Motivation

Many differential systems arising in engineering, physics, and control theory exhibit challenging features such as:

- Singular perturbations and boundary layers  
- Proportional or functional delays  
- Matrix-valued states with structural constraints  

Classical numerical methods often require problem-specific discretization strategies, dense meshes, or specialized solvers to handle these difficulties.

PINNs offer an alternative, mesh-free approach by embedding the governing physics directly into the learning process.

---

## Core Idea

The central idea of the PINN framework is to approximate the unknown solution using a neural network while enforcing the governing differential equations through the loss function.

For a generic differential equation:
\[
\mathcal{F}(y(t)) = 0,
\]
a neural network \( y_\theta(t) \) is trained by minimizing the physics residual:
\[
\mathcal{L}_{\text{phys}} = \|\mathcal{F}(y_\theta(t))\|^2.
\]

No external training data is required.

---

## Key Features

- Automatic differentiation for exact derivative computation  
- Mesh-free formulation using collocation points  
- Hard or soft enforcement of boundary and initial conditions  
- Direct handling of delayed arguments (e.g., \( y(t/2) \))  
- Natural extension to matrix-valued differential equations  

---

## Problems Addressed

The framework is demonstrated on three representative problem classes:

1. Singularly perturbed boundary value problems  
2. Pantograph-type delay differential equations  
3. Matrix Riccati differential equations from optimal control  

Each problem highlights a different advantage of the PINN methodology.

---

## Intended Use

This framework is intended for:

- Academic research  
- Methodological exploration of PINNs  
- Reproducible numerical experiments  

The emphasis is on correctness, clarity, and structure preservation rather than performance benchmarking.

---

## Limitations

- Training time is higher than classical solvers for single evaluations  
- Hyperparameter tuning may be required for more complex systems  
- PINNs do not replace classical solvers but complement them  

---

## Summary

The presented PINN framework provides a unified computational approach for solving a broad class of challenging differential systems, particularly where stiffness, delays, or matrix structure play a central role.
