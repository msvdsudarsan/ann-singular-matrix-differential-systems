
%% run_all_experiments.m
%%
%% Paper Title: "Adaptive Physics-Informed Neural Networks for Singular Matrix
%%               Differential Systems with Algebraic Structure Preservation:
%%               Applications to Optimal Control Synthesis"
%% Author 1:    Sri Venkata Durga Sudarsan Madhyannapu
%% Author 2:    Pradheep Kumar S.
%%
%% Affiliation 1: Freshmen Engineering Department,
%%                Dr. RVR NRI Institute of Technology Deemed to be University,
%%                Pothavarappadu Village, Agiripalli Mandal 521212,
%%                Vijayawada Rural, Andhra Pradesh, India
%% Affiliation 2: Research Scholar,
%%                Jawaharlal Nehru Technological University Kakinada,
%%                Andhra Pradesh, India
%% Affiliation 3: School of Basic Sciences, SRM University AP,
%%                Neerukonda, Mangalagiri, Guntur-522240, Andhra Pradesh, India
%%
%% Journal:       Engineering Applications of Artificial Intelligence
%%                (Elsevier), ISSN: 0952-1976
%% Status:        submitted 10 May 2026
%% SSRN:          https://doi.org/10.2139/ssrn.6277631
%%
%% Run this script in MATLAB R2023b (or later) to reproduce all
%% numerical experiments and tables reported in the paper.
%% Requires: Deep Learning Toolbox, Optimization Toolbox.
%%
function run_all_experiments()

    fprintf('=======================================================\n');
    fprintf('Adaptive PINN Framework for Singular Matrix Systems\n');
    fprintf('Engineering Applications of Artificial Intelligence\n');
    fprintf('Reproducing Tables 1-3 from manuscript\n');
    fprintf('=======================================================\n\n');

    %% ---------------- Problem 1 ----------------
    %  Singularly Perturbed BVP — Table 1, Fig. 1
    %  eps*y'' + y' = 0, y(0)=0, y(1)=1, eps=0.01
    %  Target: PINN MAE = (3.11+/-2.53)e-06, collocation pts = 265
    fprintf('\n>>> Problem 1: Singularly Perturbed BVP <<<\n');
    fprintf('    Domain [0,1],  eps = 0.01\n');
    fprintf('=======================================================\n');
    pinn_singular_perturbation();

    %% ---------------- Problem 2 ----------------
    %  Pantograph Delay DDE — Table 2, Fig. 2
    %  y'(t) = -y(t) + 0.5*y(0.5*t), y(0)=1, domain [0,5]
    %  Target: PINN MAE = (9.27+/-5.91)e-04
    fprintf('\n>>> Problem 2: Pantograph Delay Differential Equation <<<\n');
    fprintf('    Domain [0,5],  alpha = 0.5\n');
    fprintf('=======================================================\n');
    pinn_pantograph_delay();

    %% -- dde23 baseline (Table 2, right panel Fig. 2) --
    fprintf('\n>>> dde23 Baseline (Table 2 comparison) <<<\n');
    fprintf('=======================================================\n');
    pantograph_dde23();

    %% ---------------- Problem 3 ----------------
    %  Matrix Riccati Equation — Table 3, Fig. 3
    %  2x2 system, Cholesky-PINN + hybrid refinement
    %  Target: Standalone MAE = (1.52+/-0.18)e-01,
    %          Hybrid MAE     = (1.48+/-0.15)e-09,
    %          Symmetry error < 1e-15 (algebraic guarantee)
    fprintf('\n>>> Problem 3: Matrix Riccati Differential Equation <<<\n');
    fprintf('    2x2 system, Cholesky parameterisation (Theorem 2)\n');
    fprintf('=======================================================\n');
    pinn_matrix_riccati();

    fprintf('\n=======================================================\n');
    fprintf('ALL EXPERIMENTS COMPLETED SUCCESSFULLY\n');
    fprintf('Results correspond to single-seed runs.\n');
    fprintf('Paper values are mean+/-std over 3 independent seeds.\n');
    fprintf('=======================================================\n');
    fprintf('\nTo reproduce additional paper results, run:\n');
    fprintf('  ablation_study          --> Table 4 (ablation, all 5 configs)\n');
    fprintf('  run_robustness_30trials --> Table 5 (30-trial robustness)\n');
    fprintf('  spacecraft_riccati_6d   --> Figure 5 (6D aerospace, 50x speedup)\n');
    fprintf('=======================================================\n');

end
