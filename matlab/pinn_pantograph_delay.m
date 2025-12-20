function pinn_pantograph
% =====================================================
% Problem 2: Pantograph Equation 
% y'(t) = a y(t) + b y(ct), t ∈ [0,1]
% y(0) = 1
% =====================================================

clc; clear;

a = -1;
b = 0.5;
c = 0.5;
y0 = 1;

% reference solution (high-accuracy RK)
y_ref = @(t) reference_solution(t,a,b,c,y0);

% collocation points
t = linspace(0,1,200)';
t_dl = dlarray(t','CB');

% network
layers = [
featureInputLayer(1)
fullyConnectedLayer(50)
tanhLayer
fullyConnectedLayer(50)
tanhLayer
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

disp('Training started');

for epoch = 1:3000
[loss,grads] = dlfeval(@lossFun,net,t_dl,a,b,c,y0);
[net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,lr);

if mod(epoch,300)==0
fprintf('Epoch %d, Loss = %.3e\n',epoch,extractdata(loss));
end
end

% ---------------- Evaluation ----------------
tt = linspace(0,1,1000)';
raw = extractdata(predict(net,dlarray(tt','CB')));

% ✔ CORRECT hard IC
y_pred = y0 + tt.*raw;
y_true = y_ref(tt);

MAE = mean(abs(y_pred-y_true));
MaxErr = max(abs(y_pred-y_true));

fprintf('MAE = %.3e\n',MAE);
fprintf('Max Error = %.3e\n',MaxErr);

figure;
plot(tt,y_true,'b','LineWidth',2); hold on;
plot(tt,y_pred,'r--','LineWidth',1.5);
legend('Reference','PINN');
grid on;

end

% =====================================================
function [loss,grads] = lossFun(net,t,a,b,c,y0)

raw = forward(net,t);
y = y0 + t.*raw; % ✔ hard IC

dy = dlgradient(sum(y,'all'),t);

tc = c*t;
raw_c = forward(net,tc);
y_c = y0 + tc.*raw_c;

res = dy - a*y - b*y_c;
loss = mean(res.^2);

grads = dlgradient(loss,net.Learnables);
end

% =====================================================
function y = reference_solution(t,a,b,c,y0)

N = 6000;
tt = linspace(0,1,N);
dt = tt(2)-tt(1);
yy = zeros(1,N);
yy(1) = y0;

for k = 1:N-1
f = @(ti,yi) a*yi + b*interp1(tt,yy,c*ti,'linear',y0);
k1 = f(tt(k),yy(k));
k2 = f(tt(k)+dt/2,yy(k)+dt*k1/2);
k3 = f(tt(k)+dt/2,yy(k)+dt*k2/2);
k4 = f(tt(k)+dt,yy(k)+dt*k3);
yy(k+1) = yy(k) + dt*(k1+2*k2+2*k3+k4)/6;
end

y = interp1(tt,yy,t,'linear');
end
