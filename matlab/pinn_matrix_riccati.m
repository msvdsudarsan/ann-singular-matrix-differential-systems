%% PINN for Matrix Riccati Differential Equation (Structure-Preserving)
% dP/dt = -PA - A'P + PBR^{-1}B'P - Q
% P(1) = I
% MATLAB Online compatible

clear; clc; close all;

%% System matrices
A = [0 1; -2 -3];
B = [0; 1];
Q = eye(2);
R = 1;
T = 1;

%% Collocation points
t = linspace(0,T,1000)';
dlT = dlarray(t','CB');

%% Network (outputs lower-triangular entries)
layers = [
    featureInputLayer(1,'Normalization','none')
    fullyConnectedLayer(50)
    tanhLayer
    fullyConnectedLayer(50)
    tanhLayer
    fullyConnectedLayer(3)
];

net = dlnetwork(layers);

%% Training parameters
epochs = 15000;
lr = 1e-3;
avgGrad = [];
avgSqGrad = [];
Pbar = 0.01*eye(2);

disp('Training PINN (Problem-3)...');

for k = 1:epochs
    [loss,grad] = dlfeval(@modelLoss,net,dlT,A,B,Q,R,Pbar);
    [net,avgGrad,avgSqGrad] = adamupdate(net,grad,avgGrad,avgSqGrad,k,lr);

    if mod(k,1500)==0
        fprintf('Epoch %d | Loss %.3e\n',k,extractdata(loss));
    end
end

disp('Training completed.');

%% Evaluation
tTest = linspace(0,T,100)';
Ppinn = zeros(2,2,length(tTest));

for i = 1:length(tTest)
    dlTi = dlarray(tTest(i),'CB');
    out = extractdata(forward(net,dlTi));
    L = [out(1) 0; out(2) out(3)];
    Ppinn(:,:,i) = L*L' + Pbar;
end

%% Plot results
figure;
plot(tTest,squeeze(Ppinn(1,1,:)),'r','LineWidth',2); hold on;
plot(tTest,squeeze(Ppinn(2,2,:)),'b','LineWidth',2);
xlabel('t'); ylabel('P_{ii}(t)');
legend('P_{11}','P_{22}');
grid on;

%% ===== Loss Function =====
function [loss,gradients] = modelLoss(net,dlT,A,B,Q,R,Pbar)

loss = 0;

for i = 1:size(dlT,2)
    ti = dlT(:,i);
    out = forward(net,ti);

    L = [out(1) 0; out(2) out(3)];
    P = L*L' + Pbar;

    dP = dlgradient(sum(P,'all'),ti);

    res = dP + P*A + A'*P - P*B*(1/R)*B'*P + Q;
    loss = loss + sum(res(:).^2);
end

loss = loss/size(dlT,2);
gradients = dlgradient(loss,net.Learnables);

end
