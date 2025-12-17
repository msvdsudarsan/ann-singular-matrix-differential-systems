%% PINN for Singularly Perturbed Boundary Value Problem
% epsilon = 0.01
% Boundary layer near t = 0
% y'(t) + epsilon*y''(t) = 0
% y(0)=0, y(1)=1

clear; clc; close all;

%% Problem parameters
epsilon = 0.01;
tmin = 0;
tmax = 1;

% Exact solution (boundary layer at t = 0)
exact_solution = @(t) (1 - exp(-t/epsilon)) ./ (1 - exp(-1/epsilon));

%% Collocation points
N = 200;
t = linspace(tmin,tmax,N)';
dlT = dlarray(t','CB');

%% Neural network architecture
layers = [
    featureInputLayer(1,'Normalization','none')
    fullyConnectedLayer(30)
    tanhLayer
    fullyConnectedLayer(30)
    tanhLayer
    fullyConnectedLayer(1)
];

net = dlnetwork(layers);

%% Training parameters
numEpochs = 4000;
learningRate = 1e-3;
trailingAvg = [];
trailingAvgSq = [];

%% Training loop
disp('Training PINN for singular perturbation problem...');

for epoch = 1:numEpochs

    [loss,gradients] = dlfeval(@modelLoss,net,dlT,epsilon);

    [net,trailingAvg,trailingAvgSq] = adamupdate( ...
        net,gradients,trailingAvg,trailingAvgSq,epoch,learningRate);

    if mod(epoch,200)==0
        fprintf('Epoch %d, Loss = %.3e\n',epoch,extractdata(loss));
    end
end

%% Prediction
tTest = linspace(tmin,tmax,400)';
dlTTest = dlarray(tTest','CB');

Ntest = predict(net,dlTTest);
yPred = extractdata(dlTTest + dlTTest.*(1-dlTTest).*Ntest)';

yExact = exact_solution(tTest);

%% Error metrics
MAE = mean(abs(yPred - yExact));
relL2 = norm(yPred - yExact)/norm(yExact);

fprintf('\nMean Absolute Error (MAE) = %.3e\n',MAE);
fprintf('Relative L2 Error        = %.3e\n',relL2);

%% Plot results
figure;
plot(tTest,yExact,'k','LineWidth',2); hold on;
plot(tTest,yPred,'r--','LineWidth',2);
xlabel('t'); ylabel('y(t)');
legend('Exact','PINN','Location','Best');
title('Singularly Perturbed Boundary Value Problem (PINN)');
grid on;

%% ================= LOSS FUNCTION =================
function [loss,gradients] = modelLoss(net,dlT,epsilon)

    % Network output
    N = forward(net,dlT);

    % ✅ Correct hard boundary condition embedding
    % y(0)=0, y(1)=1
    y = dlT + dlT.*(1-dlT).*N;

    % First and second derivatives
    dy = dlgradient(sum(y,'all'),dlT,'EnableHigherDerivatives',true);
    d2y = dlgradient(sum(dy,'all'),dlT);

    % Singularly perturbed ODE residual
    % y'(t) + epsilon*y''(t) = 0
    residual = dy + epsilon*d2y;

    % Loss
    loss = mean(residual.^2);

    % Gradients
    gradients = dlgradient(loss,net.Learnables);
end
