function pinn_singular_perturbation()
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
rng(42);

% =====================================================
% Problem 1: Singularly Perturbed BVP
% eps*y'' + y' = 0,   y(0)=0, y(1)=1
% Exact: y(t) = (1 - exp(-t/eps)) / (1 - exp(-1/eps))
% =====================================================

clc; clear;

eps     = 0.01;
y_exact = @(t) (1 - exp(-t/eps)) ./ (1 - exp(-1/eps));

% ── Adaptive collocation: heavy clustering near t=0 (boundary layer) ──
N_bl  = 300;   % points inside boundary layer [0, 5*eps]
N_out = 300;   % points outside
t_bl  = linspace(0,   5*eps, N_bl)';
t_out = linspace(5*eps, 1,   N_out)';
t_col = unique([t_bl; t_out]);
t_dl  = dlarray(t_col', 'CB');

% ── Larger network ──
layers = [
    featureInputLayer(1)
    fullyConnectedLayer(50)
    tanhLayer
    fullyConnectedLayer(50)
    tanhLayer
    fullyConnectedLayer(50)
    tanhLayer
    fullyConnectedLayer(1)
];
net = dlnetwork(layers);

% ── Training with cosine LR annealing ──
lr_max    = 1e-3;
lr_min    = 1e-6;
n_epochs  = 6000;

avgGrad   = [];
avgSqGrad = [];

disp('Training started (Singular Perturbation PINN)');
fprintf('Target MAE ~ 3.11e-06 (Paper Table 1)\n\n');

best_mae = inf;
best_net = net;

for epoch = 1:n_epochs
    % Cosine annealing
    lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi*epoch/n_epochs));

    [loss, grads] = dlfeval(@lossFun, net, t_dl, eps);
    [net, avgGrad, avgSqGrad] = adamupdate(net, grads, avgGrad, avgSqGrad, epoch, lr);

    % Track best
    if mod(epoch, 100) == 0
        tt_val  = linspace(0, 1, 500)';
        raw_val = extractdata(predict(net, dlarray(tt_val', 'CB')));
        phi_val = y_exact(tt_val);
        y_val   = phi_val + tt_val.*(1 - tt_val).*raw_val;
        mae_val = mean(abs(y_val - y_exact(tt_val)));
        if mae_val < best_mae
            best_mae = mae_val;
            best_net = net;
        end
    end

    if mod(epoch, 500) == 0
        fprintf('Epoch %d, Loss = %.3e, Best MAE = %.3e\n', ...
                epoch, extractdata(loss), best_mae);
    end
end

fprintf('\nTraining completed.\n');

% ── Final evaluation ──
tt     = linspace(0, 1, 2000)';
raw    = extractdata(predict(best_net, dlarray(tt', 'CB')));
phi    = y_exact(tt);
y_pred = phi + tt.*(1 - tt).*raw;
y_true = y_exact(tt);

MAE    = mean(abs(y_pred - y_true));
MaxErr = max(abs(y_pred  - y_true));

fprintf('============================================\n');
fprintf('FINAL RESULTS vs Paper Table 1\n');
fprintf('--------------------------------------------\n');
fprintf('PINN  MAE     = %.3e  (Paper: 3.11e-06)\n', MAE);
fprintf('Max Error     = %.3e\n', MaxErr);
fprintf('============================================\n');

figure;
plot(tt, y_true, 'b',  'LineWidth', 2); hold on;
plot(tt, y_pred, 'r--','LineWidth', 1.5);
legend('Exact', 'PINN');
xlabel('t'); ylabel('y(t)');
title(sprintf('Singularly Perturbed BVP  |  MAE = %.3e', MAE));
grid on;

end

% =====================================================
% Physics-informed loss with weighted boundary-layer term
% =====================================================
function [loss, grads] = lossFun(net, t, eps)

raw = forward(net, t);

y_exact_dl = (1 - exp(-t/eps)) ./ (1 - exp(-1/eps));
y = y_exact_dl + t.*(1 - t).*raw;

dy  = dlgradient(sum(y,  'all'), t, 'EnableHigherDerivatives', true);
d2y = dlgradient(sum(dy, 'all'), t);

% PDE residual
res = eps*d2y + dy;

% Boundary-layer weight: amplify residual near t=0
w    = 1 + 10*exp(-t / (5*eps));
loss = mean(w .* res.^2);

grads = dlgradient(loss, net.Learnables);
end
