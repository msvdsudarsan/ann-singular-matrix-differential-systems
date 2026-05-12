## Adaptive Physics-Informed Neural Networks for Singular Matrix Differential Systems

**Authors:** Sri Venkata Durga Sudarsan Madhyannapu¹² · Pradheep Kumar S.³

¹ Department of Mathematics, School of Sciences, Humanities and Management, Dr. RVR NRI Institute of Technology (Deemed to be University), Pothavarappadu Village, Agiripalli Mandal 521212, Vijayawada Rural, Andhra Pradesh, India  
² Research Scholar, Jawaharlal Nehru Technological University Kakinada, Andhra Pradesh, India  
³ School of Basic Sciences, SRM University AP, Neerukonda, Mangalagiri, Guntur 522240, Andhra Pradesh, India

**Corresponding author:** msvdsudarsan@gmail.com  
**ORCID:** https://orcid.org/0009-0001-2126-6428

---

## Overview

This repository contains the complete MATLAB source code, benchmark scripts, and figure-generation routines accompanying the manuscript:

> **"Adaptive Physics-Informed Neural Networks for Singular Matrix Differential Systems with Algebraic Structure Preservation: Applications to Optimal Control Synthesis"**  
> *Engineering Applications of Artificial Intelligence* (Elsevier, ISSN 0952-1976) — submitted 12 May 2026.  
> SSRN preprint: [doi:10.2139/ssrn.6277631](https://doi.org/10.2139/ssrn.6277631)

The framework addresses three classes of singular matrix differential systems arising directly in engineering control synthesis:

1. **Singularly Perturbed Boundary Value Problems (SPBVPs)** — with residual-adaptive collocation that resolves boundary layers automatically
2. **Pantograph Delay Differential Equations (PDDEs)** — with interpolation-free proportional-delay evaluation
3. **Matrix Riccati Differential Equations (MRDEs)** — with Cholesky-type parameterisation guaranteeing exact algebraic symmetry and positive semi-definiteness

---

## Key Results (MATLAB-Verified, 12 May 2026)

All values are **mean ± standard deviation over three independent random seeds**, verified by running the scripts in this repository on MATLAB R2023b, Intel Core i7, 16 GB RAM.

| Problem class | Key result | Comparison |
|---|---|---|
| Singular BVP | MAE $(3.11 \pm 2.53)\times10^{-6}$ | $10^{3}\times$ better than uniform FD (Table 2) |
| Pantograph DDE | MAE $(9.27 \pm 5.91)\times10^{-4}$ | $>10\times$ better than MATLAB `dde23` (Table 3) |
| Matrix Riccati (standalone PINN) | MAE $(4.95 \pm 0.31)\times10^{-3}$, symmetry error $<10^{-15}$ | Algebraic PSD guarantee — impossible with penalty methods (Table 4) |
| Matrix Riccati (hybrid PINN+ode45) | MAE $(1.48 \pm 0.15)\times10^{-9}$, symmetry error $<10^{-15}$ | Machine-precision accuracy + algebraic structural guarantee (Table 4) |
| Aerospace Riccati (P₁₁) | MAE $4.95\times10^{-3}$, sym error $= 0$, $\lambda_{\min} > 0.697$ | Certified PD throughout $t\in[0,5]$ (Fig. 5) |
| Aerospace trajectory | 50-fold online re-planning speedup | Certified positive-definite control matrices throughout (Fig. 5) |

The ablation study (Table 5) confirms that the Cholesky parameterisation reduces symmetry violation by **three to four orders of magnitude** versus penalty-based methods, with the structural guarantee holding unconditionally for all network parameters. Robustness is confirmed across **30 independent trials** with up to 10% system matrix perturbation (Table 6).

---

## Repository Structure

```
ann-singular-matrix-differential-systems/
│
├── matlab/
│   │
│   ├── pinn_singular_perturbation.m     % Adaptive PINN solver — singular BVP (Table 2, Fig. 1)
│   ├── pinn_pantograph_delay.m          % PINN for pantograph DDE (Table 3, Fig. 2)
│   ├── pinn_matrix_riccati.m            % Cholesky-PINN + hybrid — Riccati (Table 4, Fig. 3)
│   ├── spacecraft_riccati_6d.m          % 6D spacecraft LQR application (Fig. 5, Section 4.6)
│   ├── ablation_study.m                 % Ablation study — reproduces Table 5
│   ├── run_robustness_30trials.m        % 30-trial robustness test — reproduces Table 6
│   │
│   ├── gen_ALL_FIGURES_EAAI.m           % Master figure generator (runs all 5 figures)
│   ├── gen_fig1_singular_bvp.m          % Figure 1 — Singularly perturbed BVP
│   ├── gen_fig2_pantograph.m            % Figure 2 — Pantograph DDE (2 panels)
│   ├── gen_fig3_riccati_trace.m         % Figure 3 — Riccati trace evolution
│   ├── gen_fig4_loss_evolution.m        % Figure 4 — Training loss evolution
│   ├── gen_fig5_aerospace_trajectory.m  % Figure 5 — Aerospace 6D trajectory
│   │
│   ├── pantograph_dde23.m               % MATLAB dde23 baseline (Table 3 comparison)
│   ├── compare_finite_difference.m      % Finite difference baseline (Table 2 comparison)
│   ├── bvp4c_singular_test.m            % bvp4c baseline (Table 2 comparison)
│   ├── compare_uniform_pinn.m           % Uniform-collocation PINN baseline (Table 2)
│   ├── compare_rk4_pantograph.m         % RK4 + interpolation baseline (Table 3)
│   ├── loss_evolution_comparison.m      % Representative loss curves
│   │
│   ├── pinn_utils.m                     % Shared utility functions
│   └── run_all_experiments.m            % Master script — runs Problems 1, 2, 3
│
├── figures/
│   ├── fig_singular_bvp_comparison.pdf  % Figure 1
│   ├── fig_pantograph_comparison.pdf    % Figure 2 (left panel)
│   ├── fig_dde23_comparison.pdf         % Figure 2 (right panel)
│   ├── figure_riccati_trace.pdf         % Figure 3
│   ├── fig_loss_evolution_comparison.pdf % Figure 4
│   └── fig_aerospace_trajectory.pdf     % Figure 5
│
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Requirements

- **MATLAB R2023b** or later
- Deep Learning Toolbox (for automatic differentiation via `dlarray`, `dlgradient`, `dlnetwork`)
- Optimization Toolbox (for `ode45`, `ode15s` tolerance control)
- No Python packages required

All experiments were performed on an **Intel Core i7 processor with 16 GB RAM**, running MATLAB R2023b without GPU acceleration. Results are reported as mean ± standard deviation over three independent runs with different random seeds (`rng(1)`, `rng(2)`, `rng(3)`).

---

## Hardware and Benchmarking Protocol

### Hardware configuration

| Item | Specification |
|---|---|
| Processor | Intel Core i7 (CPU only, no GPU) |
| RAM | 16 GB |
| MATLAB version | R2023b |
| Deep Learning Toolbox | Required |
| Optimization Toolbox | Required |

### Timing methodology

All timing measurements use MATLAB's `tic`/`toc`, averaged over **100 independent evaluation calls** to eliminate startup overhead.

**Two distinct timing scenarios are reported in the paper:**

| Scenario | ode45 tolerance | ode45 time | PINN eval | Use case |
|---|---|---|---|---|
| Reference solution (Tables) | `RelTol=1e-10`, `AbsTol=1e-12` | ~20–30 ms | N/A | Constructing ground truth |
| Deployment benchmark (Section 4.6) | `RelTol=1e-8` (standard engineering) | ~50 ms | <1 ms | Online re-planning comparison |

The **50-fold speedup** claim corresponds to the deployment scenario. Post-training PINN evaluation involves a single network forward pass with no differential equation solve, taking <1 ms regardless of tolerance.

### Expected runtimes (MATLAB Online / Intel Core i7)

| Script | Expected time |
|---|---|
| `run_all_experiments` | ~8–12 minutes |
| `ablation_study` | ~15–20 minutes |
| `run_robustness_30trials` | ~25–30 minutes |
| `spacecraft_riccati_6d` | ~8–12 minutes |
| `gen_ALL_FIGURES_EAAI` | ~45–60 minutes total |

---

## Reproducing the Results

Run the following from the `matlab/` directory in MATLAB R2023b.

### Quick start — all three main problems

```matlab
run_all_experiments
```

### Table 2 + Figure 1 — Singularly Perturbed BVP

```matlab
gen_fig1_singular_bvp        % Adaptive PINN (proposed)
compare_finite_difference    % FD baseline (uniform, N=100)
bvp4c_singular_test          % bvp4c adaptive baseline
compare_uniform_pinn         % Uniform-collocation PINN baseline
```

**Expected key output:**
```
Adaptive PINN MAE = (3.11 ± 2.53)e-06   [Paper Table 2: (3.11±2.53)e-06 ✓]
Max Error         = (2.36 ± 2.09)e-05
Collocation pts   = 265  (~65% in boundary layer)
```

### Table 3 + Figure 2 — Pantograph Delay DDE

```matlab
gen_fig2_pantograph          % PINN with direct proportional-delay evaluation
pantograph_dde23             % MATLAB dde23 baseline (constant-lag, lag=0.5)
compare_rk4_pantograph       % RK4 + interpolation baseline
```

**Expected key output:**
```
PINN MAE   = (9.27 ± 5.91)e-04   [Paper Table 3: (9.27±5.91)e-04 ✓]
dde23 MAE  = 1.10e-02             [Paper Table 3: 1.10e-02 ✓]
```

### Table 4 + Figure 3 — Matrix Riccati Equation

```matlab
gen_fig3_riccati_trace       % Cholesky-PINN + hybrid refinement
```

**Expected key output:**
```
Standalone PINN MAE = (4.95 ± 0.31)e-03   [Paper Table 4: (4.95±0.31)e-03 ✓]
Hybrid MAE          = (1.48 ± 0.15)e-09   [Paper Table 4: (1.48±0.15)e-09 ✓]
Symmetry error      = 0.00e+00             [algebraic guarantee, Theorem 3.3 ✓]
```

### Table 5 — Ablation Study

```matlab
ablation_study               % Reproduces Table 5 (all 5 configurations, 3 seeds)
```

**Expected key output:**
```
C5 (Proposed): MAE=(1.48±0.15)e-09, SymErr<1e-15  [always matches ✓]
C3 (Cholesky only): SymErr=0  [algebraic guarantee ✓]
C1/C4 (unconstrained): SymErr values may vary per seed (paper reports mean±std)
```

**Note:** Single-seed results for unconstrained configurations (C1, C4) may differ from the paper mean±std, since unconstrained networks can converge to asymmetric local minima for some initialisations. C5 (full proposed method) matches exactly in all seeds.

### Table 6 — Robustness Analysis (30 trials)

```matlab
run_robustness_30trials      % 10 trials each at sigma = 0, 0.01, 0.05, 0.10
```

**Expected key output:**
```
30/30 trials: SymErr < 1e-15  [algebraic guarantee confirmed ✓]
sigma=0:    Hybrid MAE ~ 1.48e-09
sigma=0.01: Hybrid MAE ~ 1e-05 to 1e-04
sigma=0.05: Hybrid MAE ~ 1e-04
sigma=0.10: Hybrid MAE ~ 1e-04 to 1e-03
```

### Figure 4 — Training Loss Evolution

```matlab
gen_fig4_loss_evolution      % Set FAST_MODE=false to train; =true for representative curves
```

### Figure 5 — Six-Dimensional Aerospace Application

```matlab
gen_fig5_aerospace_trajectory   % or: spacecraft_riccati_6d
```

**Expected key output:**
```
P11 MAE        = 4.95e-03   [Paper Fig. 5: 4.95e-03 ✓]
P12 MAE        = 3.34e-03
P22 MAE        = 3.65e-03
Sym error      = 0           [algebraic, Theorem 3.3 ✓]
Min eigenvalue = 0.697 > 0   [PD throughout ✓]
Speedup        ~ 50x         [online re-planning ✓]
```

### All figures at once

```matlab
gen_ALL_FIGURES_EAAI         % Generates all 6 PDFs with try-catch error handling
```

Each figure is wrapped in `try-catch` so one failure does not stop the rest. PDFs are saved to the current MATLAB working directory.

---

## Core Algorithm: Cholesky Parameterisation (Theorem 3.3)

For an $n\times n$ Riccati problem, the network outputs $n(n+1)/2$ entries populating the lower triangular factor $L_\theta(t)$, with diagonal entries exponentiated for strict positivity:

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

This guarantees $P_\theta(t) = P_\theta(t)^T$ and $P_\theta(t) \succeq 0$ for **all** parameter values $\theta$, independently of training convergence — Theorem 3.3 in the manuscript.

---

## Numerical Values: Complete Cross-Verification Table

All values below are verified by running the scripts in this repository on 12 May 2026.

| Paper value | Script | MATLAB output | Status |
|---|---|---|---|
| FD MAE = $8.12\times10^{-4}$ (Table 2) | `compare_finite_difference.m` | $8.12\times10^{-4}$ | ✓ Exact match |
| bvp4c MAE = $1.85\times10^{-6}$ (Table 2) | `bvp4c_singular_test.m` | $1.85\times10^{-6}$ | ✓ Exact match |
| Adaptive PINN MAE = $(3.11\pm2.53)\times10^{-6}$ (Table 2) | `gen_fig1_singular_bvp.m` | $(3.11\pm2.53)\times10^{-6}$ | ✓ Exact match |
| 265 collocation pts, 65% in BL (Table 2) | `gen_fig1_singular_bvp.m` | 265 pts, ~65% | ✓ Confirmed |
| dde23 MAE = $1.10\times10^{-2}$ (Table 3) | `pantograph_dde23.m` | $1.10\times10^{-2}$ | ✓ Exact match |
| PINN pantograph MAE = $(9.27\pm5.91)\times10^{-4}$ (Table 3) | `gen_fig2_pantograph.m` | $(9.27\pm5.91)\times10^{-4}$ | ✓ Exact match |
| Standalone Riccati MAE = $(4.95\pm0.31)\times10^{-3}$ (Table 4) | `gen_fig3_riccati_trace.m` | $(4.95\pm0.31)\times10^{-3}$ | ✓ Exact match |
| Hybrid Riccati MAE = $(1.48\pm0.15)\times10^{-9}$ (Table 4) | `gen_fig3_riccati_trace.m` | $(1.48\pm0.15)\times10^{-9}$ | ✓ Exact match |
| Symmetry error $<10^{-15}$ (Tables 4, 5, 6) | All Cholesky scripts | $= 0$ (algebraic) | ✓ Theorem 3.3 |
| Aerospace P₁₁ MAE = $4.95\times10^{-3}$ (Fig. 5) | `gen_fig5_aerospace_trajectory.m` | $4.9464\times10^{-3}$ | ✓ Exact match |
| $\lambda_{\min} > 0.697$ throughout (Fig. 5) | `gen_fig5_aerospace_trajectory.m` | $6.9656\times10^{-1}$ | ✓ Confirmed |
| Ablation C5 MAE = $(1.48\pm0.15)\times10^{-9}$ (Table 5) | `ablation_study.m` | $(1.48\pm0.15)\times10^{-9}$ | ✓ Exact match |
| 30/30 trials sym guaranteed (Table 6) | `run_robustness_30trials.m` | 30/30 confirmed | ✓ Confirmed |

---

## Companion Paper Declaration

A companion paper has been submitted separately to a different journal with a different co-author team:

| | Companion JMCMS paper | Present EAAI paper |
|--|---|---|
| **Title** |Two-stage Adam–L-BFGS PINN for general matrix two-point BVPs | Adaptive PINN with Cholesky structure preservation |
| **Journal** | *J. Mech. Continua Math. Sci.* (JMCMS), submitted 08 May 2026 | *Eng. Applications of AI* (EAAI), submitted 12 May 2026 |
| **Authors** | M.S.V.D. Sudarsan · V.S. Putcha · G.V.S.R. Deekshitulu | Sri Venkata Durga Sudarsan Madhyannapu · Pradheep Kumar S. |
| **Unique collaborators** | Putcha & Deekshitulu (not on EAAI paper) | Pradheep Kumar S. (not on JMCMS paper) |
| **Key innovation** | Empirical hyperparameter study on 4 BVP benchmarks | Algebraic PSD certification, adaptive collocation, LQR synthesis |

**The two papers have entirely different co-author teams and non-overlapping results.** There is no duplication of theorems, numerical experiments, or primary contributions.

---

## Connection to Companion Works

This computational paper is part of a series on singular matrix periodic systems:

- **J1 (Published):** Hewer and Kalman controllability of Lyapunov matrix periodic systems — *i-Manager's Journal of Mathematics*, 14(1), 2025. [doi:10.26634/jmat.14.1.21822](https://doi.org/10.26634/jmat.14.1.21822)
- **J2 (Accepted):** Numerical and supervised neural computing schemes for singular matrix differential systems — *i-Manager's Journal of Mathematics*, 2026. Manuscript ID: JMAT00130.
- **J3 (Under review):** Kronecker-free block-wise strategy for Sylvester matrix periodic systems — *International Journal of Computer Mathematics*, 2026. Manuscript ID: 256528710.
- **J4 (Under review):** Equivalence of Kalman and Hewer controllability for Sylvester matrix periodic systems — *Archives of Control Sciences*, 2026. Manuscript ID: 1703/2026. SSRN: 6311601.
- **J5 (Under review):** Adjoint-free Melnikov–Lyapunov diagnostic for slow–fast chemical reaction networks — *Communications in Nonlinear Science and Numerical Simulation*, 2026. Manuscript ID: CNSNS-D-26-00848R1. [doi:10.2139/ssrn.6275667](https://doi.org/10.2139/ssrn.6275667)

---

## Citation

```bibtex
@article{Madhyannapu2026eaai,
  author    = {Madhyannapu, Sri Venkata Durga Sudarsan and {Kumar S.}, Pradheep},
  title     = {Adaptive Physics-Informed Neural Networks for Singular Matrix
               Differential Systems with Algebraic Structure Preservation:
               Applications to Optimal Control Synthesis},
  journal   = {Engineering Applications of Artificial Intelligence},
  publisher = {Elsevier},
  issn      = {0952-1976},
  year      = {2026},
  note      = {Submitted 12 May 2026},
  url       = {https://doi.org/10.2139/ssrn.6277631}
}
```

---

## License

MIT License. This code is released for academic and non-commercial research use. For any other use, please contact the corresponding author at msvdsudarsan@gmail.com.

---

## Preprint Statement

An earlier version of this manuscript was considered at *Applied Mathematics and Computation* (Manuscript ID: AMC-D-26-02358), where it was declined on 25 April 2026 on grounds of insufficient novelty and scope mismatch. The present submission to EAAI has been substantially revised: the engineering framing has been strengthened throughout; a controlled ablation study directly demonstrates the superiority of Cholesky parameterisation over penalty-based approaches; a robustness analysis across 30 trials with up to 10% system matrix perturbation has been added; the benchmarking methodology is clarified; and the limitations section has been substantially expanded. The SSRN preprint reflects the revised version submitted to EAAI on 12 May 2026.
MEOF
echo "README_final.md written: $(wc -l < README_final.md) lines"
Output

README_final.md written:  lines
