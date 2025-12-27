function pinn_matrix_riccati()
% =====================================================
% Problem 3: Matrix Riccati Differential Equation
% Structure-preserving PINN (Symmetric + PD)
% =====================================================

clc; clear; close all;

% System matrices
A = [0 1; -1 -0.5];
B = [0; 1];
Q = eye(2);
R = 1;
S = eye(2);        % Terminal condition
T = 5;

% Collocation points
Nc = 150;
t = linspace(0,T,Nc)';
t_dl = dlarray(t','CB');

% Neural network (Cholesky factors)
layers = [
    featureInputLayer(1)
    fullyConnectedLayer(48)
    tanhLayer
    fullyConnectedLayer(48)
    tanhLayer
    fullyConnectedLayer(3)   % l11, l21, l22
];
net = dlnetwork(layers);

% Optimizer
lr = 1e-3;
epochs = 4000;
avgGrad = [];
avgSqGrad = [];

fprintf('Training Riccati PINN\n');

for epoch = 1:epochs
    [loss,grads] = dlfeval(@lossFun,net,t_dl,A,B,Q,R,S);
    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,lr);

    if mod(epoch,500)==0
        fprintf('Epoch %d, Loss %.3e\n',epoch,extractdata(loss));
    end
end

fprintf('Riccati training done\n');

end

% =====================================================
% Loss Function
% =====================================================
function [loss,grads] = lossFun(net,t,A,B,Q,R,S)

L = forward(net,t);

l11 = L(1,:);
l21 = L(2,:);
l22 = L(3,:);

% Symmetric + PD construction
P11 = l11.^2 + 1e-3;
P12 = l11 .* l21;
P22 = l21.^2 + l22.^2 + 1e-3;

Nc = size(t,2);
res = 0;

for k = 1:Nc
    P = [P11(k) P12(k); P12(k) P22(k)];
    dPdt = dlgradient(sum(P,'all'),t(k),'EnableHigherDerivatives',true);

    ric = -P*A - A'*P + P*B*(1/R)*(B'*P) - Q;
    res = res + sum((dPdt - ric).^2,'all');
end

% Terminal condition
PT = [P11(end) P12(end); P12(end) P22(end)];
loss_tc = sum((PT - S).^2,'all');

loss = res/Nc + 10*loss_tc;
grads = dlgradient(loss,net.Learnables);

end
