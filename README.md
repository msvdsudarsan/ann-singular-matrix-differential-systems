# ann-singular-matrix-differential-systems
Physics-Informed Neural Networks for Singular Matrix Differential Systems
Update README

# Physics-Informed Neural Networks for Singular Differential Systems

This repository contains MATLAB implementations of
Physics-Informed Neural Networks (PINNs) for solving
singular differential systems.

The focus is on numerical solution strategies rather than
benchmark optimization or dataset-driven learning.

---

## Problem Classes Covered

The implemented examples include:

- Singularly perturbed boundary value problems with boundary layers
- Delay differential equations of pantograph type
- Matrix-valued differential equations arising in control theory

These problems are known to be challenging for traditional
numerical methods due to stiffness, delay effects,
and structural constraints.

---

## Methodology

The approach follows a physics-informed learning paradigm:

- Neural networks approximate the unknown solution
- Governing differential equations are enforced through loss functions
- Derivatives are computed using automatic differentiation
- Boundary and initial conditions are embedded or penalized during training

No external training data is required.

---

## Repository Structure


---

## Usage

1. Open MATLAB (or MATLAB Online)
2. Navigate to the `matlab/` directory
3. Run the scripts individually to reproduce results

All scripts are self-contained and can be executed independently.

---

## Notes

The implementations are intended for academic
and educational purposes.

Users are encouraged to modify network size,
training parameters, and collocation strategies
to explore problem-dependent behavior.

---

## License

This repository is shared for research and academic use.
Add repository README
