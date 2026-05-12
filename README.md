## Adaptive Physics-Informed Neural Networks for Singular Matrix Differential Systems

**Authors:** Sri Venkata Durga Sudarsan Madhyannapu¹² · Pradheep Kumar S.³

¹ Department of Mathematics, School of Sciences, Humanities and Management, Dr. RVR NRI Institute of Technology (Deemed to be University), Pothavarappadu, Andhra Pradesh, India  
² Research Scholar, Jawaharlal Nehru Technological University Kakinada, Andhra Pradesh, India  
³ School of Basic Sciences, SRM University AP, Neerukonda, Mangalagiri, Guntur, Andhra Pradesh, India

**Corresponding author:** msvdsudarsan@gmail.com

---

## Overview

This repository contains the complete MATLAB source code, benchmark scripts, and figure-generation routines accompanying the manuscript:

> **"Adaptive Physics-Informed Neural Networks for Singular Matrix Differential Systems with Algebraic Structure Preservation: Applications to Optimal Control Synthesis"**  
> *Engineering Applications of Artificial Intelligence* (Elsevier, ISSN 0952-1976) — under review, 2026.

The framework addresses three classes of singular matrix differential systems that arise directly in engineering control synthesis:

1. **Singularly Perturbed Boundary Value Problems (SPBVPs)** — with residual-adaptive collocation that resolves boundary layers automatically
2. **Pantograph Delay Differential Equations (PDDEs)** — with interpolation-free proportional-delay evaluation
3. **Matrix Riccati Differential Equations (MRDEs)** — with Cholesky-type parameterisation guaranteeing exact algebraic symmetry and positive semi-definiteness

---

## Key Results

| Problem class | Key result | Comparison |
|---|---|---|
| Singular BVP | MAE $10^{-6}$ | $10^{3}\times$ better than uniform FD |
| Pantograph DDE | MAE $9.27\times10^{-4}$ | $>10\times$ better than MATLAB `dde23` |
| Matrix Riccati (hybrid) | MAE $2.17\times10^{-5}$, symmetry error $<10^{-15}$ | Algebraic PD guarantee — impossible with penalty methods |
| Aerospace trajectory | 50-fold online re-planning speedup | Certified positive-definite control matrices throughout |

The ablation study confirms that the Cholesky parameterisation reduces symmetry violation by **three to four orders of magnitude** versus penalty-based methods, with the structural guarantee holding unconditionally for all network parameters.

---

## Repository Structure

```
ann-singular-matrix-differential-systems/
│
├── matlab/
│   ├── spbvp/
│   │   ├── pinn_spbvp_adaptive.m        % Adaptive PINN solver — singular BVP
│   │   ├── spbvp_exact_solution.m       % Exact solution: (1-exp(-t/eps))/(1-exp(-1/eps))
│   │   └── run_spbvp_benchmark.m        % Reproduces Table 1 and Fig. 1
│   │
│   ├── pantograph/
│   │   ├── pinn_pantograph.m            % PINN solver — proportional-delay DDE
│   │   ├── rk_pantograph_interp.m       % Runge-Kutta with grid interpolation baseline
│   │   ├── dde23_pantograph.m           % MATLAB dde23 baseline (constant-lag)
│   │   └── run_pantograph_benchmark.m   % Reproduces Table 2 and Fig. 2
│   │
│   ├── riccati/
│   │   ├── pinn_riccati_cholesky.m      % Cholesky-PINN for matrix Riccati equation
│   │   ├── hybrid_pinn_ode45.m          % Hybrid PINN + ode45 refinement (Algorithm 2)
│   │   ├── ablation_study.m             % Reproduces Table 4 (ablation)
│   │   └── run_riccati_benchmark.m      % Reproduces Table 3 and Fig. 3
│   │
│   ├── aerospace/
│   │   ├── spacecraft_riccati_6d.m      % 6-dimensional LQR spacecraft application
│   │   └── run_aerospace_benchmark.m    % Reproduces Fig. 5 and speedup result
│   │
│   ├── robustness/
│   │   └── run_robustness_30trials.m    % 30-trial Gaussian perturbation robustness test
│   │
│   ├── hyperparameter/
│   │   └── bayesian_tuning.m            % Bayesian hyperparameter optimisation
│   │
│   └── utils/
│       ├── adaptive_collocation.m       % Algorithm 1: residual-adaptive refinement
│       ├── cholesky_output_layer.m      % Cholesky parameterisation P = L*L'
│       ├── compute_symmetry_error.m     % Frobenius-norm symmetry diagnostic
│       └── xavier_init.m               % Xavier weight initialisation
│
├── figures/
│   ├── fig_singular_bvp_comparison.pdf
│   ├── fig_pantograph_comparison.pdf
│   ├── fig_dde23_comparison.pdf
│   ├── figure_riccati_trace.pdf
│   ├── fig_loss_evolution_comparison.pdf
│   └── fig_aerospace_trajectory.pdf
│
├── submission/
│   ├── main.tex                         % Main manuscript (elsarticle, EAAI)
│   ├── cover_letter_EAAI.tex
│   ├── highlights_EAAI.tex
│   ├── consolidated_abstract_EAAI.tex
│   ├── declarations_EAAI.tex
│   ├── graphical_abstract_EAAI.tex
│   ├── suggested_reviewers_EAAI.tex
│   └── elsarticle.cls
│
└── README.md
```

---

## Requirements

- **MATLAB R2023b** or later
- Deep Learning Toolbox (for automatic differentiation)
- Optimization Toolbox (for Bayesian hyperparameter tuning)
- No additional Python packages required

All experiments were performed on an Intel Core i7 processor with 16 GB RAM. Results are reported as mean ± standard deviation over three independent runs with different random seeds.

---

## Reproducing the Results

All benchmark runs can be executed from the MATLAB command window. Results match the tables and figures in the manuscript within the reported standard deviation.

**Table 1 — Singular BVP (Fig. 1):**
```matlab
cd matlab/spbvp
run_spbvp_benchmark
```

**Table 2 — Pantograph DDE (Fig. 2):**
```matlab
cd matlab/pantograph
run_pantograph_benchmark
```

**Table 3 + Table 4 — Matrix Riccati and ablation (Fig. 3):**
```matlab
cd matlab/riccati
run_riccati_benchmark
ablation_study
```

**Fig. 5 — Six-dimensional aerospace application:**
```matlab
cd matlab/aerospace
run_aerospace_benchmark
```

**30-trial robustness test (Section 4.4):**
```matlab
cd matlab/robustness
run_robustness_30trials
```

---

## Core Algorithm: Cholesky Parameterisation

The structural certification mechanism is encapsulated in `matlab/utils/cholesky_output_layer.m`. For an $n\times n$ Riccati problem, the network outputs a vector of $n(n+1)/2$ entries that populate the lower triangular factor $L_\theta(t)$, with diagonal entries passed through `exp()` to ensure strict positivity. The output is then:

```matlab
function P = cholesky_output(net_output, n)
% Construct lower triangular L from network output
L = zeros(n, n);
idx = 1;
for col = 1:n
    for row = col:n
        if row == col
            L(row, col) = exp(net_output(idx));  % strictly positive diagonal
        else
            L(row, col) = net_output(idx);
        end
        idx = idx + 1;
    end
end
P = L * L';  % Symmetric and positive semi-definite by construction
end
```

This guarantees $P_\theta(t) = P_\theta(t)^T$ and $P_\theta(t) \succeq 0$ for **all** parameter values $\theta$, independently of training convergence — the key result of Theorem 2 in the manuscript.

---

## Connection to Companion Works

This computational paper is part of a series on singular matrix periodic systems:

- **J1 (Published):** Hewer and Kalman controllability of Lyapunov matrix periodic systems — *i-Manager's Journal of Mathematics*, 14(1), 2025. [doi:10.26634/jmat.14.1.21822](https://doi.org/10.26634/jmat.14.1.21822)
- **J2 (Accepted):** Numerical and supervised neural computing schemes for singular matrix differential systems — *i-Manager's Journal of Mathematics*, 2026.
- **J3 (Under review):** Kronecker-free block-wise strategy for Sylvester matrix periodic systems — *International Journal of Computer Mathematics*, 2026.
- **J4 (Under review):** Equivalence of Kalman and Hewer controllability for Sylvester matrix periodic systems — *Archives of Control Sciences*, 2026.
- **J5 (Under review, SSRN preprint):** Adjoint-free Melnikov–Lyapunov diagnostic for slow–fast chemical reaction networks — [doi:10.2139/ssrn.6275667](https://doi.org/10.2139/ssrn.6275667)

---

## Citation

If this code or manuscript contributes to your research, please cite:

```
M.S.V.D. Sudarsan, Pradheep Kumar S.,
"Adaptive Physics-Informed Neural Networks for Singular Matrix Differential Systems
with Algebraic Structure Preservation: Applications to Optimal Control Synthesis,"
Engineering Applications of Artificial Intelligence, Elsevier (2026), under review.
```

A preprint is available at: [doi:10.2139/ssrn.6277631](https://doi.org/10.2139/ssrn.6277631)

---

## License

This code is released for academic and non-commercial research use. For any other use, please contact the corresponding author.

---

## Preprint Statement

An earlier version of this manuscript was considered at *Applied Mathematics and Computation* (Manuscript ID: AMC-D-26-02358), where it was declined on 25 April 2026 on grounds of insufficient novelty and scope mismatch. The present submission to EAAI has been substantially revised: the engineering framing has been strengthened throughout; a controlled ablation study directly demonstrates the superiority of Cholesky parameterisation over penalty-based approaches; a robustness analysis across 30 trials with up to 10% system matrix perturbation has been added; and the limitations section has been substantially expanded. The SSRN preprint reflects the revised version submitted to EAAI.
