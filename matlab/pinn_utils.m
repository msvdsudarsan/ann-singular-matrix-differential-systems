function [t_fit, X_fit] = pinn_utils_generate_collocation(T, numPoints)
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
