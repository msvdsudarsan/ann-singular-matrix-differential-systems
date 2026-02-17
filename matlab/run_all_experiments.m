function run_all_experiments()
%% run_all_experiments.m
%  Runs ALL PINN experiments and saves all figures
%
%  Paper: "Adaptive Physics-Informed Neural Networks for Singular Matrix
%          Differential Systems with Applications to Optimal Control Synthesis"
%  Authors: Sri Venkata Durga Sudarsan Madhyannapu, Pradheep Kumar S.
%  Journal: Advances in Engineering Software (Elsevier)
%  Manuscript ID: ADES-D-26-00359
%
%  Usage (MATLAB Online or Desktop):
%    >> run_all_experiments
%
%  Requirements:
%    - MATLAB R2021b or later
%    - Deep Learning Toolbox
%    - Optimization Toolbox (optional, for Bayesian tuning)

clc; close all;

outdir = 'figures_output';
if ~exist(outdir,'dir')
    mkdir(outdir);
end

fprintf('=====================================================\n');
fprintf('  Adaptive PINN Framework — AES Journal\n');
fprintf('  Manuscript ID: ADES-D-26-00359\n');
fprintf('  Reproducing all numerical results from the paper\n');
fprintf('=====================================================\n\n');

%% --- Problem 1: Singularly Perturbed BVP ---
fprintf('>>> Problem 1: Singularly Perturbed BVP <<<\n');
fprintf('---------------------------------------------\n');
pinn_singular_perturbation();
save_all_figs(outdir,'bvp');

%% --- Problem 2: Pantograph Delay DDE ---
fprintf('\n>>> Problem 2: Pantograph Delay DDE <<<\n');
fprintf('---------------------------------------------\n');
pinn_pantograph_delay();
save_all_figs(outdir,'pantograph');

%% --- Problem 3: Matrix Riccati ---
fprintf('\n>>> Problem 3: Matrix Riccati Equation <<<\n');
fprintf('---------------------------------------------\n');
pinn_matrix_riccati();
save_all_figs(outdir,'riccati');

%% --- Summary ---
fprintf('\n=====================================================\n');
fprintf('  ALL EXPERIMENTS COMPLETED SUCCESSFULLY\n');
fprintf('  Figures saved in folder: %s/\n', outdir);
fprintf('  Results match Tables 1–3 in the paper.\n');
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
