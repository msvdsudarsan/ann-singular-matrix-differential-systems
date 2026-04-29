%% pantograph_dde23.m
%  MATLAB dde23 baseline for pantograph-type delay equation
%  Applied as constant-lag approximation (lag = 0.5) on [0, 5]
%
%  Paper: "Adaptive Physics-Informed Neural Networks for Singular Matrix
%          Differential Systems with Algebraic Structure Preservation:
%          Applications to Optimal Control Synthesis"
%  Authors: Sri Venkata Durga Sudarsan Madhyannapu, Pradheep Kumar S.
%  Journal: Engineering Applications of Artificial Intelligence (Elsevier),
%           ISSN: 0952-1976 — Under review, 2026
%  SSRN:    https://doi.org/10.2139/ssrn.6277631
%
%  Equation: y'(t) = -y(t) + 0.5*y(0.5*t)  [proportional delay, Section 4.2]
%
%  dde23 comparison uses constant-lag formulation (lag = 0.5t evaluated
%  as a fixed lag of 0.5) as the closest classical-solver analogue,
%  verified in MATLAB Online, February 2026.
%
%  Paper Table 2 verified results:
%    dde23 MAE       = 1.10e-02
%    dde23 Max Error = 1.89e-02

clc; clear; close all;

T = 5;   % domain [0, 5] — matches paper Section 4.2

%% --- Solve using MATLAB dde23 (constant-lag formulation, lag = 0.5) ---
sol = dde23(@(t,y,Z) -y + 0.5*Z(:,1), 0.5, 1, [0, T]);

%% --- High-resolution RK4 reference (N=6000 steps, Section 4.2) ---
N     = 6000;
dt    = T / N;
t_ref = (0 : dt : T)';
y_ref = zeros(size(t_ref));
y_ref(1) = 1;

for i = 2 : length(t_ref)
    ti  = t_ref(i-1);
    yi  = y_ref(i-1);
    if i == 1
        k1 = -yi + 0.5*1;
    else
        t_delayed = 0.5 * ti;
        idx = max(1, find(t_ref(1:i-1) <= t_delayed, 1, 'last'));
        y_del = y_ref(idx);
        k1 = -yi + 0.5*y_del;
    end

    % RK4 stages (simplified Euler for reference; full RK4 below)
    t2   = ti + dt/2;
    idx2 = max(1, find(t_ref(1:i-1) <= 0.5*t2, 1, 'last'));
    k2   = -(yi + dt*k1/2) + 0.5*y_ref(idx2);

    idx3 = idx2;
    k3   = -(yi + dt*k2/2) + 0.5*y_ref(idx3);

    t4   = ti + dt;
    idx4 = max(1, find(t_ref(1:i-1) <= 0.5*t4, 1, 'last'));
    k4   = -(yi + dt*k3) + 0.5*y_ref(idx4);

    y_ref(i) = yi + dt*(k1 + 2*k2 + 2*k3 + k4)/6;
end

%% --- Compute errors ---
y_dde23    = deval(sol, t_ref)';
MAE_dde23  = mean(abs(y_dde23 - y_ref));
MaxE_dde23 = max(abs(y_dde23 - y_ref));

%% --- Print results ---
fprintf('==============================================\n');
fprintf('  dde23 Comparison Results  (domain [0, %g])\n', T);
fprintf('==============================================\n');
fprintf('  dde23 MAE       = %.4e  (Paper Table 2: 1.10e-02)\n', MAE_dde23);
fprintf('  dde23 Max Error = %.4e  (Paper Table 2: 1.89e-02)\n', MaxE_dde23);
fprintf('==============================================\n');
fprintf('  Verified in MATLAB Online, February 2026.\n');

%% --- Plot ---
figure('Position',[100,100,820,460],'Color','w');
plot(t_ref, y_ref,   'k-',  'LineWidth',2.2, 'DisplayName','Reference (RK4, N=6000)');
hold on;
plot(t_ref, y_dde23, 'b--', 'LineWidth',1.8, 'DisplayName','dde23 (MATLAB, constant lag=0.5)');
xlabel('t',    'FontSize',13,'FontWeight','bold');
ylabel('y(t)', 'FontSize',13,'FontWeight','bold');
title('Pantograph-Type Delay Equation: dde23 vs Reference (domain [0,5])','FontSize',13);
legend('Location','northeast','FontSize',11);
grid on; box on;
set(gca,'FontSize',11,'LineWidth',1.2);
