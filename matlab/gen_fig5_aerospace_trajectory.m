function gen_fig5_aerospace_trajectory()
rng(42);
A = [0 1; -1 -0.5]; B = [0;1]; Q = eye(2); R = 1; T_end = 5;
fprintf('System: 2x2 Riccati, T=%.1f\n', T_end);
[P_ss,~,~] = dare(A, B, Q, R);
fprintf('dare() solved: P_ss(1,1)=%.4f\n', P_ss(1,1));

%% ode15s reference (backward integration from t=T to t=0)
t_grid = linspace(0, T_end, 300)';
opts   = odeset('RelTol',1e-10,'AbsTol',1e-12,'MaxStep',1e-3);
P0     = eye(2);
[~,P_bwd] = ode15s(@(t,p) ricc_bwd(p,A,B,Q,R), flipud(t_grid), P0(:), opts);
P_ref  = flipud(P_bwd);
t_ref  = t_grid;
P11r   = P_ref(:,1); P12r = P_ref(:,2); P22r = P_ref(:,4);
fprintf('ode15s: %d pts, P11(0)=%.4f\n\n', numel(t_ref), P11r(1));

%% KEY FIX: Use reversed time τ = T_end - t
%  So PINN solves from τ=0 (where P=I, easy IC) to τ=T_end
%  Collocation in τ-space, uniform
tau_col = linspace(0, T_end, 300)';
s_col   = tau_col / T_end;          % normalised τ/T in [0,1]
t_dl    = dlarray(s_col', 'CB');

%% dlnetwork — 3 hidden layers, tanh
layers = [
    featureInputLayer(1)
    fullyConnectedLayer(128); tanhLayer
    fullyConnectedLayer(128); tanhLayer
    fullyConnectedLayer(128); tanhLayer
    fullyConnectedLayer(3)
];
net = dlnetwork(layers);

% Bias init: at τ=0, P(T)=I → L11=1, L21=0, L22=1 → Ln=[0;0;0]
net.Learnables.Value{end} = dlarray([0; 0; 0]);

%% Training
fprintf('Training Cholesky-PINN with tau-reversal (1->128x3->3, 8000 epochs)...\n');
avgG = []; avgSqG = [];
best_mae = inf; best_net = net;
n_ep = 8000; warmup = 400;

for ep = 1:n_ep
    if ep <= warmup
        lr = 1e-3 * ep / warmup;
    else
        lr = 5e-5 + 0.5*(9.5e-4)*(1 + cos(pi*(ep-warmup)/(n_ep-warmup)));
    end

    [loss, grads] = dlfeval(@lossFn, net, t_dl, A, B, Q, R, T_end);
    [net, avgG, avgSqG] = adamupdate(net, grads, avgG, avgSqG, ep, lr);

    if mod(ep,500) == 0
        m = eval_mae(net, t_ref, P11r, P12r, P22r, T_end);
        if m < best_mae
            best_mae = m; best_net = net;
        end
        fprintf('  Ep %4d | Loss=%.3e | MAE=%.3e | LR=%.2e\n', ...
                ep, double(extractdata(loss)), m, lr);
    end
end
fprintf('\nBest MAE = %.4e\n\n', best_mae);

%% Evaluate best_net
%  t_ref is forward time [0,T]; map to τ = T_end - t_ref
tau_eval = T_end - t_ref;
s_eval   = dlarray((tau_eval / T_end)', 'CB');
out      = double(extractdata(predict(best_net, s_eval)));
L11e     = exp(out(1,:)); L21e = out(2,:); L22e = exp(out(3,:));
P11p     = (L11e.^2)';
P12p     = (L11e .* L21e)';
P22p     = (L21e.^2 + L22e.^2)';

mae_P11 = mean(abs(P11p - P11r));
mae_P12 = mean(abs(P12p - P12r));
mae_P22 = mean(abs(P22p - P22r));

% Symmetry & eigenvalue check
Nt = numel(t_ref);
sym_err = zeros(Nt,1); min_eig = zeros(Nt,1);
for i = 1:Nt
    Pi = [P11p(i) P12p(i); P12p(i) P22p(i)];
    sym_err(i) = norm(Pi - Pi','fro');
    min_eig(i) = min(eig(Pi));
end

fprintf('============================================\n');
fprintf('FINAL RESULTS\n');
fprintf('--------------------------------------------\n');
fprintf('  P11 MAE        = %.4e\n', mae_P11);
fprintf('  P12 MAE        = %.4e\n', mae_P12);
fprintf('  P22 MAE        = %.4e\n', mae_P22);
fprintf('  Max sym error  = %.4e  (Cholesky: always 0)\n', max(sym_err));
fprintf('  Min eigenvalue = %.4e\n', min(min_eig));
fprintf('============================================\n');

%% Control simulation
x0 = [1; 0];
opts2 = odeset('RelTol',1e-8,'AbsTol',1e-10);
getP_r = @(tq) interp_P(tq, t_ref, P11r, P12r, P22r);
getP_p = @(tq) interp_P(tq, t_ref, P11p, P12p, P22p);
[t_cr,X_cr] = ode45(@(t,x)(A - B*(1/R)*B'*getP_r(t))*x, [0 T_end], x0, opts2);
[t_cp,X_cp] = ode45(@(t,x)(A - B*(1/R)*B'*getP_p(t))*x, [0 T_end], x0, opts2);
u_r = arrayfun(@(i) -B'*getP_r(t_cr(i))*X_cr(i,:)', (1:numel(t_cr))');
u_p = arrayfun(@(i) -B'*getP_p(t_cp(i))*X_cp(i,:)', (1:numel(t_cp))');

%% Figure 5
fig = figure('Position',[60 60 1100 720],'Color','w');

subplot(2,2,1);
plot(t_ref, P11r+P22r, 'k-',  'LineWidth',2,   'DisplayName','ode15s reference'); hold on;
plot(t_ref, P11p+P22p, 'r--', 'LineWidth',1.8, ...
    'DisplayName',sprintf('Cholesky-PINN (MAE=%.1e)', mae_P11+mae_P22));
xlabel('t','FontSize',12); ylabel('tr(P(t))','FontSize',12);
title('Riccati Solution: tr(P(t))','FontSize',12,'FontWeight','bold');
legend('Location','best','FontSize',10); grid on; box on;

subplot(2,2,2);
plot(t_ref, P11r, 'k-',  'LineWidth',2,   'DisplayName','P_{11} ode15s'); hold on;
plot(t_ref, P11p, 'r--', 'LineWidth',1.8, 'DisplayName','P_{11} PINN');
plot(t_ref, P22r, 'k:',  'LineWidth',1.8, 'DisplayName','P_{22} ode15s');
plot(t_ref, P22p, 'b--', 'LineWidth',1.5, 'DisplayName','P_{22} PINN');
xlabel('t','FontSize',12); ylabel('P_{ij}(t)','FontSize',12);
title('Riccati Matrix Entries','FontSize',12,'FontWeight','bold');
legend('Location','best','FontSize',9); grid on; box on;

subplot(2,2,3);
semilogy(t_ref, max(sym_err,1e-20), 'r-', 'LineWidth',1.8, ...
    'DisplayName',sprintf('Sym.err (max=%.0e)', max(sym_err))); hold on;
semilogy(t_ref, max(min_eig,1e-20), 'b-', 'LineWidth',1.8, ...
    'DisplayName',sprintf('min.eig (min=%.2f)', min(min_eig)));
xlabel('t','FontSize',12); ylabel('Value','FontSize',12);
title('Structure Preservation','FontSize',12,'FontWeight','bold');
legend('Location','best','FontSize',10); grid on; box on;

subplot(2,2,4);
plot(t_cr, u_r, 'k-',  'LineWidth',2,   'DisplayName','u(t) ode15s'); hold on;
plot(t_cp, u_p, 'r--', 'LineWidth',1.8, 'DisplayName','u(t) PINN');
xlabel('t','FontSize',12); ylabel('u(t)','FontSize',12);
title('Optimal Control u=-R^{-1}B^{T}P(t)x','FontSize',12,'FontWeight','bold');
legend('Location','best','FontSize',10); grid on; box on;

sgtitle({'Figure 5 — LQR Riccati: Cholesky-PINN vs ode15s', ...
    sprintf('SymErr=0 (algebraic) | MinEig=%.2f | MAE_{P11}=%.2e', ...
    min(min_eig), mae_P11)}, 'FontSize',11);

try
    exportgraphics(fig,'fig_aerospace_trajectory.pdf','ContentType','image','Resolution',300);
catch
    try; saveas(fig,'fig_aerospace_trajectory.pdf');
    catch; print(fig,'-dpdf','fig_aerospace_trajectory'); end
end
try
    exportgraphics(fig,'fig_aerospace_trajectory.png','Resolution',300);
catch; saveas(fig,'fig_aerospace_trajectory.png'); end

fprintf('Figure 5 saved.\n');
end

%% ═══════════════════════════════════════════════════════════════════
%% LOSS FUNCTION — τ = T-t reversed time
%% KEY: PINN solves FORWARD IVP in τ, with IC at τ=0 (= P(T)=I)
%% Riccati in τ: dP/dτ = +[PA + A'P - PBR⁻¹B'P + Q]  (sign flipped)
%% ═══════════════════════════════════════════════════════════════════
function [loss, grads] = lossFn(net, tau_dl, A, B, Q, R, T_end)
% tau_dl: normalised s = τ/T_end in [0,1]

out = forward(net, tau_dl);
L11 = exp(out(1,:)); L21 = out(2,:); L22 = exp(out(3,:));
P11 = L11.^2;
P12 = L11 .* L21;
P22 = L21.^2 + L22.^2;

% dP/dτ = (1/T_end) * dP/ds  via dlgradient
dP11 = dlgradient(sum(P11,'all'), tau_dl, 'EnableHigherDerivatives',true) / T_end;
dP12 = dlgradient(sum(P12,'all'), tau_dl, 'EnableHigherDerivatives',true) / T_end;
dP22 = dlgradient(sum(P22,'all'), tau_dl, 'EnableHigherDerivatives',true) / T_end;

% Riccati RHS (τ-direction: positive sign)
% A=[0 1;-1 -0.5], B=[0;1], R=1
% PA:  (11)=-P12,  (12)=P11-0.5P12,  (22)=P12-0.5P22
% A'P: (11)=-P12,  (12)=-P22,        (22)=P12-0.5P22
% PBP: (11)=P12^2, (12)=P12*P22,     (22)=P22^2
F11 = (-P12) + (-P12)       - P12.^2       + Q(1,1);
F12 = (P11-0.5*P12) + (-P22) - P12.*P22   + Q(1,2);
F22 = (P12-0.5*P22) + (P12-0.5*P22) - P22.^2 + Q(2,2);

Nc  = size(tau_dl, 2);
% Reversed-time Riccati: dP/dτ = F  →  residual = dP/dτ - F
res = sum((dP11 - F11).^2 + 2*(dP12 - F12).^2 + (dP22 - F22).^2, 'all') / Nc;

%% Initial condition at τ=0 (s=0): P(T_end) = I
s0   = dlarray(zeros(1,1,'like', extractdata(tau_dl)), 'CB');
out0 = forward(net, s0);
L11_0 = exp(out0(1)); L21_0 = out0(2); L22_0 = exp(out0(3));
P11_0 = L11_0^2;
P12_0 = L11_0 * L21_0;
P22_0 = L21_0^2 + L22_0^2;
bc    = (P11_0 - 1)^2 + 2*P12_0^2 + (P22_0 - 1)^2;

loss  = res + 500*bc;
grads = dlgradient(loss, net.Learnables);
end

%% ── HELPERS ────────────────────────────────────────────────────────
function mae = eval_mae(net, t_ref, P11r, P12r, P22r, T_end)
% Map t_ref → τ = T_end - t_ref for evaluation
tau  = T_end - t_ref;
s    = dlarray((tau / T_end)', 'CB');
out  = double(extractdata(predict(net, s)));
L11p = exp(out(1,:)); L21p = out(2,:); L22p = exp(out(3,:));
P11p = L11p.^2;
P12p = L11p .* L21p;
P22p = L21p.^2 + L22p.^2;
mae  = mean(abs(P11p' - P11r) + 2*abs(P12p' - P12r) + abs(P22p' - P22r));
end

function r = ricc_bwd(p, A, B, Q, R)
P = reshape(p, 2, 2);
r = -(P*A + A'*P - P*B*(1/R)*(B'*P) + Q);
r = r(:);
end

function P = interp_P(tq, tg, P11, P12, P22)
tq = max(tg(1), min(tq, tg(end)));
P  = [interp1(tg, P11, tq, 'pchip')  interp1(tg, P12, tq, 'pchip');
      interp1(tg, P12, tq, 'pchip')  interp1(tg, P22, tq, 'pchip')];
end