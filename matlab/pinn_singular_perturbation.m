function pinn_singular_perturbation
% =====================================================
% Problem 1 
% eps*y'' + y' = 0, y(0)=0, y(1)=1
% =====================================================

clc; clear;

eps = 0.01;

y_exact = @(t) (1-exp(-t/eps))./(1-exp(-1/eps));

% collocation points (boundary-layer aware)
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

disp('Training started');

for epoch = 1:3000
    [loss,grads] = dlfeval(@lossFun,net,t_dl,eps);
    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,lr);

    if mod(epoch,200)==0
        fprintf('Epoch %d, Loss = %.3e\n',epoch,extractdata(loss));
    end
end

% evaluation
tt = linspace(0,1,1000)';
raw = extractdata(predict(net,dlarray(tt','CB')));

% ===== CORRECT HARD BC =====
phi = (1-exp(-tt/eps))./(1-exp(-1/eps));
y_pred = phi + tt.*(1-tt).*raw;
y_true = y_exact(tt);

fprintf('MAE = %.3e\n',mean(abs(y_pred-y_true)));
fprintf('Max Error = %.3e\n',max(abs(y_pred-y_true)));

figure;
plot(tt,y_true,'b','LineWidth',2); hold on;
plot(tt,y_pred,'r--','LineWidth',1.5);
legend('Exact','PINN'); grid on;

end

% =====================================================
function [loss,grads] = lossFun(net,t,eps)

raw = forward(net,t);

phi = (1-exp(-t/eps))./(1-exp(-1/eps));
y = phi + t.*(1-t).*raw;

dy  = dlgradient(sum(y,'all'),t,'EnableHigherDerivatives',true);
d2y = dlgradient(sum(dy,'all'),t);

res = eps*d2y + dy;
loss = mean(res.^2);

grads = dlgradient(loss,net.Learnables);

end
