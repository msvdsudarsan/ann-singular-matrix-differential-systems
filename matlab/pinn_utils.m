function [t_fit, X_fit] = pinn_utils_generate_collocation(T, numPoints)
%% Paper Title: "Adaptive Physics-Informed Neural Networks for Singular Matrix
%%               Differential Systems with Applications to Optimal Control Synthesis"
%% Author 1:    Sri Venkata Durga Sudarsan Madhyannapu
%% Author 2:    Pradheep Kumar S.
%%
%% Affiliation 1: Freshmen Engineering Department, NRI Institute of Technology
%%                (Autonomous), Pothavarappadu, Agiripalli, Eluru District 521212,
%%                Andhra Pradesh, India
%% Affiliation 2: Research Scholar, Jawaharlal Nehru Technological University
%%                Kakinada, Andhra Pradesh, India
%% Affiliation 3: School of Basic Sciences, SRM University AP, Neerukonda,
%%                Mangalagiri, Guntur-522240, Andhra Pradesh, India
%%
%% Journal:       Neurocomputing (Elsevier), ISSN: 0925-2312
%% Manuscript ID: NEUCOM-D-26-03849
%% Status:        With Editor, 2026
%% SSRN:          https://ssrn.com/abstract=6277631
%% SSRN ID:       6277631 (Distributed: 02/20/2026)
%%
% Generate collocation points uniformly
t_fit = linspace(0, T, numPoints)';
% If needed, add random or Chebyshev nodes here
X_fit = t_fit; % same shape for simple 1D problems
end

function save_results(problemName, results)
% Save results to results folder
fname = sprintf('results_%s.mat', problemName);
if ~exist('results','dir')
    mkdir('results');
end
save(fullfile('results',fname),'results');
end

function val = structural_MAE(true_sol, pred_sol)
% Mean absolute error over solution
val = mean(abs(true_sol - pred_sol),'all');
end
