function pinn_singular_perturbation()

clc; clear;

numRuns = 3;
MAE_all = zeros(numRuns,1);
MaxErr_all = zeros(numRuns,1);

for seed = 1:numRuns
rng(seed);

eps = 0.01;
y_exact = @(t) (1-exp(-t/eps))./(1-exp(-1/eps));

t = linspace(0,1,200)';
t = t.^2;
t_dl = dlarray(t','CB');

layers = [
    featureInputLayer(1)
    fullyConnectedLayer(30)
    tanhLayer
    fullyConnectedLayer(30)
    tanhLayer
    fullyConnectedLayer(1)
];

net = dlnetwork(layers);

lr = 1e-3;
avgGrad = [];
avgSqGrad = [];

for epoch = 1:3000
    [loss,grads] = dlfeval(@lossFun,net,t_dl,eps);
    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,lr);
end

tt = linspace(0,1,1000)';
raw = extractdata(predict(net,dlarray(tt','CB')));

phi = (1-exp(-tt/eps))./(1-exp(-1/eps));
y_pred = phi + tt.*(1-tt).*raw;
y_true = y_exact(tt);

MAE_all(seed)    = mean(abs(y_pred - y_true),'all');
MaxErr_all(seed) = max(abs(y_pred - y_true),[],'all');

end

fprintf('MAE = %.3e ± %.3e\n', mean(MAE_all), std(MAE_all));
fprintf('Max Error = %.3e ± %.3e\n', mean(MaxErr_all), std(MaxErr_all));

% ===== FIGURE (ONLY FIX) =====
figure;
plot(tt, y_true, 'k-', 'LineWidth', 2); hold on;
plot(tt, y_pred, 'r--', 'LineWidth', 2);
xlabel('t'); ylabel('y(t)');
legend('Exact','Adaptive PINN','Location','best');
title('Singularly Perturbed BVP (\epsilon = 0.01)');
grid on;
saveas(gcf,'fig_singular_bvp_comparison.pdf');

end

function [loss,grads] = lossFun(net,t,eps)

raw = forward(net,t);
phi = (1-exp(-t/eps))./(1-exp(-1/eps));
y = phi + t.*(1-t).*raw;

dy  = dlgradient(sum(y,'all'),t,'EnableHigherDerivatives',true);
d2y = dlgradient(sum(dy,'all'),t);

res = eps*d2y + dy;
loss = mean(res.^2,'all');
grads = dlgradient(loss,net.Learnables);

end
