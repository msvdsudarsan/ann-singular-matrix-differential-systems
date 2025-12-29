function pinn_pantograph_delay()

clc; clear; close all;

numRuns = 3;
MAE_all = zeros(numRuns,1);
MaxErr_all = zeros(numRuns,1);

for seed = 1:numRuns
rng(seed);

% =====================================================
% Pantograph Delay Differential Equation
% y'(t) = a y(t) + b y(alpha t),   y(0) = 1
% =====================================================

a = -1;
b = 0.5;
alpha = 0.5;
y0 = 1;

t = linspace(0,1,400)';
t_dl = dlarray(t','CB');

layers = [
    featureInputLayer(1)
    fullyConnectedLayer(50)
    tanhLayer
    fullyConnectedLayer(50)
    tanhLayer
    fullyConnectedLayer(1)
];
net = dlnetwork(layers);

lr = 1e-3;
avgGrad = [];
avgSqGrad = [];

for epoch = 1:4000
    [loss,grads] = dlfeval(@lossFun,net,t_dl,a,b,alpha,y0);
    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,lr);
end

% ================= Evaluation =================
tt = linspace(0,1,1000)';
raw = extractdata(predict(net,dlarray(tt','CB')));

y_pred = y0 + tt .* raw;
y_true = reference_solution(tt,a,b,alpha,y0);

MAE_all(seed)    = mean(abs(y_pred - y_true),'all');
MaxErr_all(seed) = max(abs(y_pred - y_true),[],'all');

end

fprintf('MAE = %.3e ± %.3e\n', mean(MAE_all), std(MAE_all));
fprintf('Max Error = %.3e ± %.3e\n', mean(MaxErr_all), std(MaxErr_all));

end

% =====================================================
function [loss,grads] = lossFun(net,t,a,b,alpha,y0)

raw = forward(net,t);
y = y0 + t .* raw;

dy = dlgradient(sum(y,'all'),t);

tc = alpha * t;
raw_c = forward(net,tc);
y_c = y0 + tc .* raw_c;

res = dy - a*y - b*y_c;
loss = mean(res.^2,'all');

grads = dlgradient(loss,net.Learnables);
end

% =====================================================
function y = reference_solution(t,a,b,alpha,y0)

N = 6000;
tt = linspace(0,1,N);
dt = tt(2)-tt(1);
yy = zeros(1,N);
yy(1) = y0;

for k = 1:N-1
    f = @(ti,yi) a*yi + b*interp1(tt,yy,alpha*ti,'linear',y0);
    k1 = f(tt(k),yy(k));
    k2 = f(tt(k)+dt/2,yy(k)+dt*k1/2);
    k3 = f(tt(k)+dt/2,yy(k)+dt*k2/2);
    k4 = f(tt(k)+dt,yy(k)+dt*k3);
    yy(k+1) = yy(k) + dt*(k1+2*k2+2*k3+k4)/6;
end

y = interp1(tt,yy,t,'linear');
end
