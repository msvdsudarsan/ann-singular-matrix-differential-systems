function run_all_experiments()
%% Paper Title: "Adaptive Physics-Informed Neural Networks for Singular Matrix
%%               Differential Systems with Applications to Optimal Control
%%               Synthesis"
%% Author 1:    Sri Venkata Durga Sudarsan Madhyannapu
%% Author 2:    Pradheep Kumar S.
%%
%% Affiliation 1: Freshmen Engineering Department, NRI Institute of Technology,
%%                Pothavarappadu, Agiripalli, Eluru District 521212,
%%                Andhra Pradesh, India
%% Affiliation 2: Research Scholar, Jawaharlal Nehru Technological University
%%                Kakinada, Andhra Pradesh, India
%% Affiliation 3: School of Basic Sciences, SRM University AP, Neerukonda,
%%                Mangalagiri, Guntur-522240, Andhra Pradesh, India
%%
%% Journal:       Neurocomputing, Elsevier, ISSN: 0925-2312
%% Manuscript ID: NEUCOM-D-26-03849
%% Status:        Under Review, 2026

%% run_all_experiments.m
%
% Runs ALL PINN experiments and saves all figures
%
% Paper: "Adaptive Physics-Informed Neural Networks for Singular Matrix
%         Differential Systems with Applications to Optimal Control Synthesis"
% Authors: Sri Venkata Durga Sudarsan Madhyannapu, Pradheep Kumar S.
% Journal: Neurocomputing (Elsevier)
% Manuscript ID: NEUCOM-D-26-03849
% Submitted: 27 February 2026
%
% Usage (MATLAB R2023b or later):
%   >> run_all_experiments
%
% Requirements:
%   - MATLAB R2023b or later
%   - Deep Learning Toolbox
%   - Optimization Toolbox (optional, for Bayesian tuning)

clc; close all;

outdir = 'figures_output';
if ~exist(outdir,'dir'), mkdir(outdir); end

fprintf('=====================================================\n');
fprintf('  Adaptive PINN Framework\n');
fprintf('  Journal: Neurocomputing (Elsevier)\n');
fprintf('  Manuscript ID: NEUCOM-D-26-03849\n');
fprintf('  Reproducing all numerical results from the paper\n');
fprintf('=====================================================\n\n');

%% --- Problem 1: Singularly Perturbed BVP (Table 1, Fig. 1) ---
fprintf('>>> Problem 1: Singularly Perturbed BVP <<<\n');
fprintf('---------------------------------------------\n');
pinn_singular_perturbation();
save_all_figs(outdir,'bvp');

%% --- Problem 2: Pantograph Delay DDE (Table 2, Fig. 2) ---
fprintf('\n>>> Problem 2: Pantograph Delay DDE <<<\n');
fprintf('---------------------------------------------\n');
pinn_pantograph_delay();
save_all_figs(outdir,'pantograph');

%% --- Problem 3: Matrix Riccati (Table 3, Fig. 3) ---
fprintf('\n>>> Problem 3: Matrix Riccati Equation <<<\n');
fprintf('---------------------------------------------\n');
pinn_matrix_riccati();
save_all_figs(outdir,'riccati');

%% --- Summary ---
fprintf('\n=====================================================\n');
fprintf('  ALL EXPERIMENTS COMPLETED SUCCESSFULLY\n');
fprintf('  Figures saved in folder: %s/\n', outdir);
fprintf('  Results reproduce Tables 1-3 in the paper.\n');
fprintf('=====================================================\n');
end

%% =====================================================
function save_all_figs(outdir, prefix)
figs = findall(0,'Type','figure');
for k = 1:length(figs)
    timestamp = string(datetime('now','Format','yyyyMMdd'));
    fname     = sprintf('%s_%s_fig%d', timestamp, prefix, k);
    saveas(figs(k), fullfile(outdir,[fname '.pdf']));
    saveas(figs(k), fullfile(outdir,[fname '.png']));
    fprintf('  Saved: %s.pdf\n', fname);
end
close(figs);
end
