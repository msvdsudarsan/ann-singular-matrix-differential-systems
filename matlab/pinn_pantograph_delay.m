function pinn_pantograph_delay()
% =====================================================
% Problem 2: Pantograph Delay Differential Equation
% y'(t) = a y(t) + b y(alpha t),   y(0) = 1
% =====================================================

clc; clear; close all;

% Parameters
a = -1;
b = 0.5;
alpha = 0.5;
y0 = 1;

% Collocation points
t = linspace(0,1,400)';
t_dl = dlarray(t','CB');

% Network
layers = [
    featureInputLayer(1)
    fullyConnectedLayer(50)
    tanhLayer
    fullyConnectedLayer(50)
    tanhLayer
    fullyConnectedLayer(1)
];
net = dlnetwork(layers);

% Optimizer
lr = 1e-3;
avgGrad = [];
avgSqGrad = [];

fprintf('Training started (Pantograph PINN)\n');

% Training loop
for epoch = 1:4000
    [loss,grads] = dlfeval(@lossFun,net,t_dl,a,b,alpha,y0);
    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,lr);

    if mod(epoch,500)==0
        fprintf('Epoch %d, Loss = %.3e\n',epoch,extractdata(loss));
    end
end

fprintf('Pantograph training completed\n');

% ================= Evaluation =================
tt = linspace(0,1,1000)';
raw = extractdata(predict(net,dlarray(tt','CB')));

% ✅ CORRECT HARD INITIAL CONDITION (PDF MATCH)
y_pred = y0 + tt .* raw;

% Reference solution (RK4)
y_true = reference_solution(tt,a,b,alpha,y0);

MAE = mean(abs(y_pred - y_true));
MaxErr = max(abs(y_pred - y_true));

fprintf('MAE = %.3e\n',MAE);
fprintf('Max Error = %.3e\n',MaxErr);

figure;
plot(tt,y_true,'b','LineWidth',2); hold on;
plot(tt,y_pred,'r--','LineWidth',1.5);
legend('Reference','PINN');
xlabel('t'); ylabel('y(t)');
grid on;

end

% =====================================================
% Loss Function
% =====================================================
function [loss,grads] = lossFun(net,t,a,b,alpha,y0)

raw = forward(net,t);

% ✅ Correct hard IC
y = y0 + t .* raw;

% First derivative
dy = dlgradient(sum(y,'all'),t);

% Delayed term
tc = alpha * t;
raw_c = forward(net,tc);
y_c = y0 + tc .* raw_c;

% Residual
res = dy - a*y - b*y_c;

loss = mean(res.^2);

grads = dlgradient(loss,net.Learnables);
end

% =====================================================
% Reference solution (high-resolution RK4)
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
