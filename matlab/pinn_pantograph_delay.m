%% PINN for Pantograph Delay Differential Equation
% y'(t) = y(t) + y(t/2), y(0)=1

clear; clc; close all;

T = 1;
alpha = 0.5;
exact = @(t) exp(3*t/2);

t = linspace(0,T,300)';
dlT = dlarray(t','CB');

layers = [
    featureInputLayer(1,'Normalization','none')
    fullyConnectedLayer(40)
    tanhLayer
    fullyConnectedLayer(40)
    tanhLayer
    fullyConnectedLayer(1)
];

net = dlnetwork(layers);

epochs = 8000;
lr = 1e-3;
avgG = []; avgSq = [];

disp('Training PINN (Problem-2)...');

for k = 1:epochs
    [loss,grad] = dlfeval(@lossPantograph,net,dlT,alpha);
    [net,avgG,avgSq] = adamupdate(net,grad,avgG,avgSq,k,lr);
    if mod(k,1000)==0
        fprintf('Epoch %d | Loss %.3e\n',k,extractdata(loss));
    end
end

tTest = linspace(0,T,1000)';
dlTTest = dlarray(tTest','CB');
yPred = extractdata(predict(net,dlTTest))';
yTrue = exact(tTest);

fprintf('\nMAE = %.2e\n',mean(abs(yPred-yTrue)));
fprintf('Max Error = %.2e\n',max(abs(yPred-yTrue)));

figure;
plot(tTest,yTrue,'k','LineWidth',2); hold on;
plot(tTest,yPred,'r--','LineWidth',2);
legend('Exact','PINN'); grid on;

function [loss,grad] = lossPantograph(net,dlT,alpha)
    y = forward(net,dlT);
    yDelay = forward(net,alpha*dlT);
    dy = dlgradient(sum(y,'all'),dlT);
    res = dy - y - yDelay;
    loss = mean(res.^2);
    grad = dlgradient(loss,net.Learnables);
end
