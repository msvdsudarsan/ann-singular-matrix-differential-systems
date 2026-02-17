%% pantograph_dde23.m
%  MATLAB dde23 solver comparison for pantograph-type delay equation
%
%  Paper: "Adaptive Physics-Informed Neural Networks for Singular Matrix
%          Differential Systems with Applications to Optimal Control Synthesis"
%  Authors: Sri Venkata Durga Sudarsan Madhyannapu, Pradheep Kumar S.
%  Journal: Advances in Engineering Software (Elsevier)
%  Manuscript ID: ADES-D-26-00359
%
%  Equation: y'(t) = -y(t) + 0.5*y(t - 0.5),  y(t)=1 for t<=0
%
%  Verified Results (MATLAB Online, Feb 2026):
%    dde23 MAE       = 1.0951e-02
%    dde23 Max Error = 1.8948e-02

clc; clear; close all;

%% --- Solve using MATLAB dde23 ---
%  lag = 0.5 (constant delay)
sol = dde23(@(t,y,Z) -y + 0.5*Z(:,1), 0.5, 1, [0,1]);

%% --- High-resolution RK4 reference ---
dt    = 1e-4;
t_ref = 0 : dt : 1;
y_ref = zeros(size(t_ref));
y_ref(1) = 1;

for i = 2 : length(t_ref)
    ti  = t_ref(i-1);
    idx = max(1, round(0.5*ti/dt) + 1);
    idx = min(idx, i-1);
    k1  = -y_ref(i-1) + 0.5*y_ref(idx);
    y_ref(i) = y_ref(i-1) + dt*k1;
end

%% --- Compute errors ---
y_dde23    = deval(sol, t_ref);
MAE_dde23  = mean(abs(y_dde23 - y_ref));
MaxE_dde23 = max(abs(y_dde23 - y_ref));

%% --- Print results ---
fprintf('==============================================\n');
fprintf('  dde23 Comparison Results\n');
fprintf('==============================================\n');
fprintf('  dde23 MAE       = %.4e\n', MAE_dde23);
fprintf('  dde23 Max Error = %.4e\n', MaxE_dde23);
fprintf('==============================================\n');

%% --- Plot ---
figure('Position',[100,100,820,460],'Color','w');
plot(t_ref, y_ref,   'k-',  'LineWidth',2.2, 'DisplayName','Reference (RK4)');
hold on;
plot(t_ref, y_dde23, 'b--', 'LineWidth',1.8, 'DisplayName','dde23 (MATLAB)');
xlabel('t',    'FontSize',13,'FontWeight','bold');
ylabel('y(t)', 'FontSize',13,'FontWeight','bold');
title('Pantograph-Type Delay Equation: dde23 vs Reference','FontSize',14);
legend('Location','northeast','FontSize',11);
grid on; box on;
set(gca,'FontSize',11,'LineWidth',1.2);
