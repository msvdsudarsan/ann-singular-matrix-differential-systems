function pinn_matrix_riccati()
% =====================================================
% Problem 3: Matrix Riccati Differential Equation (2x2)
% Structure-preserving PINN
% P'(t) = -P A - A' P + P B R^{-1} B' P - Q
% P(T) = S
% =====================================================

clc; clear; close all;

% ---------- System matrices ----------
A = [0 1; -1 -0.5];
B = [0; 1];
Q = eye(2);
R = 1;
S = eye(2);          % terminal condition
T = 5;

% ---------- Collocation points ----------
N = 150;
t = linspace(0,T,N)';
t_dl = dlarray(t','CB');

% ---------- Network (outputs 3 values: Cholesky factors) ----------
layers = [
    featureInputLayer(1)
    fullyConnectedLayer(48)
    tanhLayer
    fullyConnectedLayer(48)
    tanhLayer
    fullyConnectedLayer(3)   % l11, l21, l22
];
net = dlnetwork(layers);

% ---------- Optimizer ----------
lr = 1e-3;
epochs = 8000;
avgGrad = [];
avgSqGrad = [];

fprintf('Training started (Structure-preserving Riccati PINN)\n');

for epoch = 1:epochs
    [loss,grads] = dlfeval(@lossFun,net,t_dl,A,B,Q,R,S,T);
    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,lr);

    if mod(epoch,500)==0
        fprintf('Epoch %d, Loss = %.3e\n',epoch,extractdata(loss));
    end
end

fprintf('Training completed\n');

% ---------- Evaluation ----------
tt = linspace(0,T,200)';
raw = extractdata(predict(net,dlarray(tt','CB')));

P11 = raw(1,:).^2 + 0.01;
P12 = raw(1,:) .* raw(2,:);
P22 = raw(2,:).^2 + raw(3,:).^2 + 0.01;

figure;
plot(tt,P11,'r','LineWidth',2); hold on;
plot(tt,P22,'b','LineWidth',2);
xlabel('t'); ylabel('P(t)');
legend('P_{11}','P_{22}');
title('Structure-Preserving PINN Riccati Solution');
grid on;

end

% =====================================================
% Loss Function
% =====================================================
function [loss,grads] = lossFun(net,t,A,B,Q,R,S,T)

L = forward(net,t);

l11 = L(1,:);
l21 = L(2,:);
l22 = L(3,:);

% Structure-preserving construction
P11 = l11.^2 + 0.01;
P12 = l11 .* l21;
P22 = l21.^2 + l22.^2 + 0.01;

N = length(t);
res = 0;

for k = 1:N
    P = [P11(k) P12(k); P12(k) P22(k)];
    dPdt = dlgradient(sum(P,'all'), t(k));

    ric = -P*A - A'*P + P*B*(1/R)*(B'*P) - Q;
    res = res + norm(dPdt - ric,'fro')^2;
end

% Terminal condition
PT = [P11(end) P12(end); P12(end) P22(end)];
loss_tc = norm(PT - S,'fro')^2;

loss = res/N + 10*loss_tc;
grads = dlgradient(loss,net.Learnables);

end
