function pinn_pantograph_delay()
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

%% pinn_pantograph_delay.m
%
% PINN for Pantograph Delay Differential Equation
%
% Paper: "Adaptive Physics-Informed Neural Networks for Singular Matrix
%         Differential Systems with Applications to Optimal Control Synthesis"
% Authors: Sri Venkata Durga Sudarsan Madhyannapu, Pradheep Kumar S.
% Journal: Neurocomputing (Elsevier)
% Manuscript ID: NEUCOM-D-26-03849
% Submitted: 27 February 2026
%
% Problem: y'(t) = -y(t) + 0.5*y(0.5*t),  y(0) = 1
%          a=-1, b=0.5, alpha=0.5 (proportional delay)
%
% Key advantage: proportional delay y(alpha*t) evaluated DIRECTLY as
% network input — no interpolation, no accumulated error.
%
% Architecture: 3 hidden layers x 50 neurons (tanh), as per paper Sec. 2.1
% Training: Phase 1 — 2000 epochs lr=1e-3; Phase 2 — 2000 epochs lr=1e-4
% Reproduces Table 2 of the paper.

clc; clear;

%% --- Parameters (paper Sec. 3.2) ---
a     = -1;
b     =  0.5;
alpha =  0.5;
y0    =  1;

numRuns    = 3;
MAE_all    = zeros(numRuns,1);
MaxErr_all = zeros(numRuns,1);

%% --- High-resolution RK4 reference (N=6000, paper Sec. 3.2) ---
tt_ref = linspace(0,1,1000)';
y_ref  = reference_solution(tt_ref, a, b, alpha, y0);

for seed = 1:numRuns
    rng(seed);

    %% --- Collocation ---
    Ncol = 200;
    tcol    = linspace(0,1,Ncol)';
    t_dl    = dlarray(tcol','CB');

    % Network: 3 hidden layers x 50 neurons (matches paper Sec. 2.1)
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

    lr        = 1e-3;
    avgGrad   = [];
    avgSqGrad = [];

    % Phase 1: 2000 epochs (paper Sec. 2.6)
    for epoch = 1:2000
        [loss,grads] = dlfeval(@lossFun, net, t_dl, a, b, alpha, y0);
        [net,avgGrad,avgSqGrad] = adamupdate(net, grads, ...
            avgGrad, avgSqGrad, epoch, lr);
    end

    % Phase 2: fine-tune 2000 more epochs (paper Sec. 2.6)
    lr = 1e-4;
    for epoch = 2001:4000
        [loss,grads] = dlfeval(@lossFun, net, t_dl, a, b, alpha, y0);
        [net,avgGrad,avgSqGrad] = adamupdate(net, grads, ...
            avgGrad, avgSqGrad, epoch, lr);
    end

    %% --- Evaluation ---
    t_eval  = dlarray(tt_ref','CB');
    raw     = extractdata(predict(net, t_eval));
    % Hard IC enforcement: y(0)=y0
    y_pred  = y0 + tt_ref .* raw(:);

    MAE_all(seed)    = mean(abs(y_pred - y_ref));
    MaxErr_all(seed) = max(abs(y_pred  - y_ref));
end

%% --- Baselines ---
fprintf('\n==============================================\n');
fprintf('  Running dde23 baseline (lag = 0.5) ...\n');
fprintf('  (Constant-lag formulation, paper Table 2)\n');

dde_rhs  = @(t,y,Z) a*y + b*Z;
delay_fn = @(t,y) t - 0.5;
y0_hist  = @(t) y0;
opts_dde = ddeset('RelTol',1e-6,'AbsTol',1e-8);
sol_dde  = dde23(dde_rhs, delay_fn, y0_hist, [0,1], opts_dde);
y_dde23  = deval(sol_dde, tt_ref')';
MAE_dde23  = mean(abs(y_dde23 - y_ref));
MaxE_dde23 = max(abs(y_dde23  - y_ref));

y_rk4      = reference_solution_rk4_coarse(tt_ref, a, b, alpha, y0);
MAE_rk4    = mean(abs(y_rk4 - y_ref));
MaxE_rk4   = max(abs(y_rk4  - y_ref));

%% --- Print results (reproduces Table 2) ---
fprintf('\n==============================================\n');
fprintf('  Pantograph DDE — Table 2\n');
fprintf('  Journal: Neurocomputing, NEUCOM-D-26-03849\n');
fprintf('==============================================\n');
fprintf('  %-25s %12s %12s\n','Method','MAE','Max Error');
fprintf('  %-25s %12.3e %12.3e\n','RK4 + interpolation', ...
    MAE_rk4, MaxE_rk4);
fprintf('  %-25s %12.3e %12.3e\n','dde23 (MATLAB)', ...
    MAE_dde23, MaxE_dde23);
fprintf('  %-25s %12.3e %12.3e\n','PINN (proposed)', ...
    mean(MAE_all), mean(MaxErr_all));
fprintf('==============================================\n');

%% --- Figure (paper Fig. 2) ---
y_pred_plot = y0 + tt_ref .* ...
    extractdata(predict(net, dlarray(tt_ref','CB')))';

figure('Position',[100,100,820,460],'Color','w');
plot(tt_ref, y_ref,       'k-',  'LineWidth',2.2, ...
    'DisplayName','Reference (High-res RK4)');
hold on;
plot(tt_ref, y_pred_plot, 'r--', 'LineWidth',1.8, ...
    'DisplayName','PINN (proposed)');
plot(tt_ref, y_dde23,     'b:',  'LineWidth',1.8, ...
    'DisplayName','dde23 (MATLAB)');
plot(tt_ref, y_rk4,       'g-.', 'LineWidth',1.6, ...
    'DisplayName','RK4 + interpolation');
xlabel('t','FontSize',13,'FontWeight','bold');
ylabel('y(t)','FontSize',13,'FontWeight','bold');
title('Pantograph Delay Differential Equation','FontSize',14);
legend('Location','northeast','FontSize',11);
grid on; box on;
set(gca,'FontSize',11,'LineWidth',1.2);
saveas(gcf,'fig_pantograph_comparison.pdf');
saveas(gcf,'fig_pantograph_comparison.png');
fprintf('\nFigure saved: fig_pantograph_comparison.pdf\n');
end

%% =====================================================
function [loss,grads] = lossFun(net, t, a, b, alpha, y0)
raw   = forward(net, t);
y     = y0 + t .* raw;
dy    = dlgradient(sum(y,'all'), t);
% Proportional delay: feed alpha*t directly into network (no interpolation)
tc    = alpha * t;
raw_c = forward(net, tc);
y_c   = y0 + tc .* raw_c;
res   = dy - a*y - b*y_c;
loss  = mean(res.^2,'all');
grads = dlgradient(loss, net.Learnables);
end

%% =====================================================
function y = reference_solution(t, a, b, alpha, y0)
% High-resolution RK4 reference (N=6000, paper Sec. 3.2)
N  = 6000;
tt = linspace(0,1,N)';
dt = tt(2) - tt(1);
yy = zeros(N,1);  yy(1) = y0;
for k = 1:N-1
    f  = @(ti,yi) a*yi + b*interp1(tt,yy,alpha*ti,'linear',y0);
    k1 = f(tt(k),         yy(k));
    k2 = f(tt(k)+dt/2,    yy(k)+dt*k1/2);
    k3 = f(tt(k)+dt/2,    yy(k)+dt*k2/2);
    k4 = f(tt(k)+dt,      yy(k)+dt*k3);
    yy(k+1) = yy(k) + dt*(k1+2*k2+2*k3+k4)/6;
end
y = interp1(tt, yy, t, 'linear');
end

%% =====================================================
function y = reference_solution_rk4_coarse(t, a, b, alpha, y0)
% Coarser RK4 (N=200) — RK4+interpolation baseline (paper Table 2)
N  = 200;
tt = linspace(0,1,N)';
dt = tt(2) - tt(1);
yy = zeros(N,1);  yy(1) = y0;
for k = 1:N-1
    f  = @(ti,yi) a*yi + ...
        b*interp1(tt,yy,alpha*ti,'linear',y0,'extrap');
    k1 = f(tt(k),      yy(k));
    k2 = f(tt(k)+dt/2, yy(k)+dt*k1/2);
    k3 = f(tt(k)+dt/2, yy(k)+dt*k2/2);
    k4 = f(tt(k)+dt,   yy(k)+dt*k3);
    yy(k+1) = yy(k) + dt*(k1+2*k2+2*k3+k4)/6;
end
y = interp1(tt, yy, t, 'linear');
end
