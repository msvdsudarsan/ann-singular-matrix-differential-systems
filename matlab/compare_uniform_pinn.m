% compare_uniform_pinn.m
% Uniform PINN baseline (NO adaptive refinement)

clear; clc;

eps = 0.01;
Ncol = 100;
tcol = linspace(0,1,Ncol)';

y_exact = @(t) (1 - exp(-t/eps)) ./ (1 - exp(-1/eps));

layers = [
    featureInputLayer(1)
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(1)
];

net = dlnetwork(layers);

numEpochs = 3000;
learnRate = 1e-3;

tcol_dl = dlarray(tcol','CB');

for epoch = 1:numEpochs
    [loss,grads] = dlfeval(@modelLoss,net,tcol_dl,eps);
    net = dlupdate(@(p,g) p - learnRate*g, net, grads);
end

% --- Evaluation (FIXED) ---
t_test = linspace(0,1,200)';
t_test_dl = dlarray(t_test','CB');

y_pred_dl = forward(net,t_test_dl);
y_pred = extractdata(y_pred_dl)';

y_ex = y_exact(t_test);

abs_err = abs(y_pred - y_ex);
MAE = mean(abs_err);
MaxErr = max(abs_err);

fprintf('Uniform PINN MAE = %.3e\n', MAE);
fprintf('Uniform PINN Max Error = %.3e\n', MaxErr);

% ---------- Loss ----------
function [loss,gradients] = modelLoss(net,t,eps)
    y = forward(net,t);
    dy = dlgradient(sum(y,'all'),t,'EnableHigherDerivatives',true);
    d2y = dlgradient(sum(dy,'all'),t);

    res = eps*d2y + dy;
    lossPDE = mean(res.^2,'all');

    y0 = forward(net,dlarray(0,'CB'));
    y1 = forward(net,dlarray(1,'CB'));
    lossBC = (y0)^2 + (y1-1)^2;

    loss = lossPDE + lossBC;
    gradients = dlgradient(loss,net.Learnables);
end
