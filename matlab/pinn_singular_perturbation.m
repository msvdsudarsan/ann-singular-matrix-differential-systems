%% PINN for Singularly Perturbed Boundary Value Problem
% ε y''(t) + y'(t) = 0
% y(0)=0, y(1)=1, ε=0.01
% MATLAB Online compatible (stable)

clear; clc; close all;

%% Parameters
epsilon = 0.01;
tmin = 0; 
tmax = 1;

exact = @(t) (1 - exp(-t/epsilon)) ./ (1 - exp(-1/epsilon));

%% Collocation points
N = 200;
t = linspace(tmin,tmax,N)';
dlT = dlarray(t','CB');

%% Network
layers = [
    featureInputLayer(1,'Normalization','none')
    fullyConnectedLayer(30)
    tanhLayer
    fullyConnectedLayer(30)
    tanhLayer
    fullyConnectedLayer(1)
];

net = dlnetwork(layers);

%% Training setup
epochs = 6000;
lr = 1e-3;
avgGrad = [];
avgSqGrad = [];

disp('Training PINN (Problem-1)...');

for k = 1:epochs
    [loss,grad] = dlfeval(@modelLoss,net,dlT,epsilon);
    [net,avgGrad,avgSqGrad] = adamupdate(net,grad,avgGrad,avgSqGrad,k,lr);

    if mod(k,1000)==0
        fprintf('Epoch %d | Loss %.3e\n',k,extractdata(loss));
    end
end

%% Evaluation
tTest = linspace(0,1,400)';
dlTTest = dlarray(tTest','CB');

Nout = predict(net,dlTTest);
yPred = extractdata(dlTTest + dlTTest.*(1-dlTTest).*Nout)';
yTrue = exact(tTest);

MAE = mean(abs(yPred - yTrue));
MaxErr = max(abs(yPred - yTrue));

fprintf('\n=== FINAL RESULTS ===\n');
fprintf('MAE       = %.2e\n',MAE);
fprintf('Max Error = %.2e\n',MaxErr);

%% Plot
figure;
plot(tTest,yTrue,'k','LineWidth',2); hold on;
plot(tTest,yPred,'r--','LineWidth',2);
legend('Exact','PINN','Location','Best');
xlabel('t'); ylabel('y(t)');
grid on;

%% Loss function
function [loss,gradients] = modelLoss(net,dlT,epsilon)
    N = forward(net,dlT);

    % Hard BC: y(0)=0, y(1)=1
    y = dlT + dlT.*(1-dlT).*N;

    dy = dlgradient(sum(y,'all'),dlT,'EnableHigherDerivatives',true);
    d2y = dlgradient(sum(dy,'all'),dlT);

    residual = dy + epsilon*d2y;
    loss = mean(residual.^2);

    gradients = dlgradient(loss,net.Learnables);
end
