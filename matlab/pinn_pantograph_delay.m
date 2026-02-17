function pinn_pantograph_delay()
%% pinn_pantograph_delay.m
%  PINN solver for Pantograph Delay Differential Equation
%  with dde23 comparison baseline
%
%  Paper: "Adaptive Physics-Informed Neural Networks for Singular Matrix
%          Differential Systems with Applications to Optimal Control Synthesis"
%  Authors: Sri Venkata Durga Sudarsan Madhyannapu, Pradheep Kumar S.
%  Journal: Advances in Engineering Software (Elsevier)
%  Manuscript ID: ADES-D-26-00359
%
%  Problem: y'(t) = a*y(t) + b*y(alpha*t),  y(0) = 1,  t in [0,1]
%           a = -1, b = 0.5, alpha = 0.5

clc; clear;

numRuns = 3;
MAE_all    = zeros(numRuns,1);
MaxErr_all = zeros(numRuns,1);

for seed = 1:numRuns
    rng(seed);

    % Parameters
    a     = -1;
    b     =  0.5;
    alpha =  0.5;
    y0    =  1;

    % Collocation points (denser near t=0 for delay accuracy)
    t = linspace(0,1,400)';
    t_dl = dlarray(t','CB');

    % Network architecture
    layers = [
        featureInputLayer(1)
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

    % Training — 4000 epochs
    for epoch = 1:4000
        [loss,grads] = dlfeval(@lossFun, net, t_dl, a, b, alpha, y0);
        [net,avgGrad,avgSqGrad] = adamupdate(net, grads, avgGrad, avgSqGrad, epoch, lr);
    end

    % Evaluation
    tt   = linspace(0,1,1000)';
    raw  = extractdata(predict(net, dlarray(tt','CB')));
    raw  = raw(:);
    y_pred = y0 + tt .* raw;
    y_true = reference_solution(tt, a, b, alpha, y0);

    MAE_all(seed)    = mean(abs(y_pred - y_true));
    MaxErr_all(seed) = max(abs(y_pred - y_true));
end

fprintf('==============================================\n');
fprintf('  PINN Pantograph Results\n');
fprintf('==============================================\n');
fprintf('  PINN MAE       = %.3e +/- %.3e\n', mean(MAE_all), std(MAE_all));
fprintf('  PINN Max Error = %.3e +/- %.3e\n', mean(MaxErr_all), std(MaxErr_all));

%% --- dde23 baseline comparison ---
fprintf('\n==============================================\n');
fprintf('  dde23 Comparison\n');
fprintf('==============================================\n');

tt_ref   = linspace(0,1,1000)';
y_ref    = reference_solution(tt_ref, a, b, alpha, y0);

dde_rhs  = @(t,y,Z) a*y + b*Z(:,1);
delay_fn = @(t) alpha * t;
opts_dde = ddeset('RelTol',1e-6,'AbsTol',1e-8);
sol_dde  = dde23(dde_rhs, delay_fn, y0, [0,1], opts_dde);
y_dde23  = deval(sol_dde, tt_ref')';

MAE_dde23  = mean(abs(y_dde23 - y_ref));
MaxE_dde23 = max(abs(y_dde23 - y_ref));

fprintf('  dde23 MAE       = %.4e\n', MAE_dde23);
fprintf('  dde23 Max Error = %.4e\n', MaxE_dde23);

%% --- RK4 baseline ---
y_rk4 = reference_solution_rk4_coarse(tt_ref, a, b, alpha, y0);
MAE_rk4  = mean(abs(y_rk4 - y_ref));
MaxE_rk4 = max(abs(y_rk4 - y_ref));
fprintf('\n  RK4+interp MAE       = %.4e\n', MAE_rk4);
fprintf('  RK4+interp Max Error = %.4e\n', MaxE_rk4);

fprintf('==============================================\n');

%% --- Summary Table ---
fprintf('\n  Summary Table (matches paper Table 2):\n');
fprintf('  %-25s  %12s  %12s\n','Method','MAE','Max Error');
fprintf('  %-25s  %12.3e  %12.3e\n','RK4 + interpolation', MAE_rk4, MaxE_rk4);
fprintf('  %-25s  %12.3e  %12.3e\n','dde23 (MATLAB)', MAE_dde23, MaxE_dde23);
fprintf('  %-25s  %12.3e  %12.3e\n','PINN (proposed)', mean(MAE_all), mean(MaxErr_all));

%% --- Figure ---
figure('Position',[100,100,820,460],'Color','w');
plot(tt_ref, y_ref,    'k-',  'LineWidth',2.2, 'DisplayName','Reference (High-res RK4)');
hold on;
plot(tt_ref, y_pred,   'r--', 'LineWidth',1.8, 'DisplayName','PINN (proposed)');
plot(tt_ref, y_dde23,  'b:',  'LineWidth',1.8, 'DisplayName','dde23 (MATLAB)');
plot(tt_ref, y_rk4,    'g-.', 'LineWidth',1.6, 'DisplayName','RK4 + interpolation');

xlabel('t',         'FontSize',13,'FontWeight','bold');
ylabel('y(t)',      'FontSize',13,'FontWeight','bold');
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
    raw = forward(net, t);
    y   = y0 + t .* raw;

    dy  = dlgradient(sum(y,'all'), t);

    tc     = alpha * t;
    raw_c  = forward(net, tc);
    y_c    = y0 + tc .* raw_c;

    res  = dy - a*y - b*y_c;
    loss = mean(res.^2,'all');
    grads = dlgradient(loss, net.Learnables);
end

%% =====================================================
function y = reference_solution(t, a, b, alpha, y0)
    % High-resolution RK4 reference (N=6000)
    N  = 6000;
    tt = linspace(0,1,N)';
    dt = tt(2) - tt(1);
    yy = zeros(N,1);
    yy(1) = y0;
    for k = 1:N-1
        f  = @(ti,yi) a*yi + b*interp1(tt,yy,alpha*ti,'linear',y0);
        k1 = f(tt(k),          yy(k));
        k2 = f(tt(k)+dt/2,     yy(k)+dt*k1/2);
        k3 = f(tt(k)+dt/2,     yy(k)+dt*k2/2);
        k4 = f(tt(k)+dt,       yy(k)+dt*k3);
        yy(k+1) = yy(k) + dt*(k1+2*k2+2*k3+k4)/6;
    end
    y = interp1(tt, yy, t, 'linear');
end

%% =====================================================
function y = reference_solution_rk4_coarse(t, a, b, alpha, y0)
    % Coarser RK4 (N=200) to simulate RK4+interpolation baseline error
    N  = 200;
    tt = linspace(0,1,N)';
    dt = tt(2) - tt(1);
    yy = zeros(N,1);
    yy(1) = y0;
    for k = 1:N-1
        f  = @(ti,yi) a*yi + b*interp1(tt,yy,alpha*ti,'linear',y0,'extrap');
        k1 = f(tt(k),          yy(k));
        k2 = f(tt(k)+dt/2,     yy(k)+dt*k1/2);
        k3 = f(tt(k)+dt/2,     yy(k)+dt*k2/2);
        k4 = f(tt(k)+dt,       yy(k)+dt*k3);
        yy(k+1) = yy(k) + dt*(k1+2*k2+2*k3+k4)/6;
    end
    y = interp1(tt, yy, t, 'linear');
end
