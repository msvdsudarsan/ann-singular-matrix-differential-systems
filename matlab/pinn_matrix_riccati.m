function pinn_matrix_riccati()
clc; clear; close all;

A = [0 1; -1 -0.5];
B = [0;1];
Q = eye(2);
R = 1;
S = eye(2);
T = 5;

t = linspace(0,T,150)';
t_dl = dlarray(t','CB');

layers = [
    featureInputLayer(1)
    fullyConnectedLayer(40)
    tanhLayer
    fullyConnectedLayer(40)
    tanhLayer
    fullyConnectedLayer(4)
];
net = dlnetwork(layers);

lr = 1e-3;
avgGrad = [];
avgSqGrad = [];

fprintf('Training Riccati PINN\n');

for epoch = 1:4000
    [loss,grads] = dlfeval(@lossFun,net,t_dl,A,B,Q,R,S,T);
    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,lr);

    if mod(epoch,500)==0
        fprintf('Epoch %d, Loss %.3e\n',epoch,extractdata(loss));
    end
end

fprintf('Riccati training done\n');
end

% =====================================================
function [loss,grads] = lossFun(net,t,A,B,Q,R,S,T)

raw = forward(net,t);

res = 0;
N = size(t,2);

for k = 1:N
    P = reshape(raw(:,k),2,2);
    dPdt = dlgradient(sum(P,'all'),t(k));

    ric = -P*A - A'*P + P*B*(1/R)*(B'*P) - Q;
    res = res + sum((dPdt-ric).^2,'all');
end

PT = reshape(raw(:,end),2,2);
loss = res/N + 10*sum((PT-S).^2,'all');

grads = dlgradient(loss,net.Learnables);
end
