# ann-singular-matrix-differential-systems

This repository presents MATLAB implementations of Physics-Informed Neural Networks (PINNs) for the numerical solution of singular matrix differential systems.

The emphasis is on physics-driven computational modeling rather than data-driven training or benchmark-oriented optimization.

---

## Problem Classes Covered

The repository includes PINN-based solvers for the following classes of problems:

- Singularly perturbed boundary value problems exhibiting boundary layer behavior  
- Pantograph-type delay differential equations  
- Matrix-valued differential equations arising in control and dynamical systems  

These problem classes are well known to be challenging for classical numerical schemes due to stiffness, delay effects, and structural constraints.

---

## Methodological Overview

The implemented framework follows a physics-informed learning strategy:

- Neural networks are used to approximate the unknown solution functions  
- Governing differential equations are enforced directly through residual-based loss functions  
- Derivatives are computed using automatic differentiation  
- Boundary and initial conditions are imposed either through penalty terms or analytical embedding  
- No external training data is required  

This approach yields mesh-free solvers that adapt naturally to stiff dynamics and complex solution structures.

---

## Repository Structure

ann-singular-matrix-differential-systems/
├── matlab/
│ ├── pinn_singular_perturbation.m
│ ├── pinn_pantograph_delay.m
│ ├── pinn_matrix_riccati.m
│ ├── pinn_utils.m
│ └── run_all_experiments.m
├── results/
├── docs/
└── README.md                                                                                                                              
---

## Usage Instructions

1. Open MATLAB or MATLAB Online  
2. Navigate to the `matlab/` directory  
3. Run the scripts individually to reproduce numerical experiments  

Each script is self-contained and can be executed independently.

---

## Notes

- The implementations are intended for academic and research use  
- Users are encouraged to modify network size, training parameters, and collocation strategies  
- The focus is on numerical behavior rather than performance benchmarking  

---

## License

This repository is shared for research and academic purposes.
