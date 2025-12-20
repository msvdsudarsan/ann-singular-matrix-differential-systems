function pinn_riccati
% =====================================================
% Problem 3: Matrix Riccati Differential Equation
% X'(t) = A'X + XA - XBR^{-1}B'X + Q
% X(0) = X0
% =====================================================

clc; clear;

% ---------- System matrices ----------
A = [0 1; -2 -3];
B = [0; 1];
Q = eye(2);
R = 1;
X0 = eye(2);

T = 1;

% ---------- Collocation points ----------
t = linspace(0,T,200)';
t_dl = dlarray(t','CB');

% ---------- Network ----------
layers = [
    featureInputLayer(1)
    fullyConnectedLayer(60)
    tanhLayer
    fullyConnectedLayer(60)
    tanhLayer
    fullyConnectedLayer(60)
    tanhLayer
    fullyConnectedLayer(4)   % 2x2 matrix → 4 outputs
];
net = dlnetwork(layers);

% ---------- Optimizer ----------
lr = 1e-3;
avgGrad = [];
avgSqGrad = [];

disp('Training started');

for epoch = 1:4000
    [loss,grads] = dlfeval(@lossFun,net,t_dl,A,B,Q,R,X0);
    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,lr);

    if mod(epoch,500)==0
        fprintf('Epoch %d, Loss = %.3e\n',epoch,extractdata(loss));
    end
end

% ---------- Evaluation ----------
tt = linspace(0,T,300)';
raw = extractdata(predict(net,dlarray(tt','CB')));

X_pred = zeros(2,2,length(tt));
for k = 1:length(tt)
    Y = reshape(raw(:,k),2,2);
    X_pred(:,:,k) = X0 + tt(k)*Y;   % hard IC
end

fprintf('Training completed\n');

end

% =====================================================
function [loss,grads] = lossFun(net,t,A,B,Q,R,X0)

raw = forward(net,t);

N = size(t,2);
residual = 0;

for k = 1:N
    Y = reshape(raw(:,k),2,2);
    X = X0 + t(k)*Y;   % hard IC

    dXdt = Y;

    ric = A'*X + X*A - X*B*(R\B')*X + Q;
    residual = residual + sum((dXdt - ric).^2,'all');
end

loss = residual / N;
grads = dlgradient(loss,net.Learnables);

end
