function pinn_pantograph_delay()
% =====================================================
% Problem 2: Pantograph Delay Differential Equation
% y'(t) = -y(t) + 0.5 y(0.5t) + sin(t),  y(0)=1
% =====================================================

clc; clear; close all;

y0 = 1;
T = 5;

% Collocation points
t = linspace(0,T,200)';
t_dl = dlarray(t','CB');

% Network
layers = [
    featureInputLayer(1)
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
epochs = 5000;
avgGrad = [];
avgSqGrad = [];

disp('Training started (Pantograph PINN)');

for epoch = 1:epochs
    [loss,grads] = dlfeval(@lossFun,net,t_dl,y0);
    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,lr);

    if mod(epoch,500)==0
        fprintf('Epoch %d, Loss = %.3e\n',epoch,extractdata(loss));
    end
end

disp('Training completed');

% -------- Evaluation --------
tt = linspace(0,T,800)';
y_pred = extractdata(predict(net,dlarray(tt','CB')));

% Reference (dde23)
sol = dde23(@dde_rhs,0.5,y0,[0 T]);
y_ref = deval(sol,tt)';

MAE = mean(abs(y_pred - y_ref));
MaxErr = max(abs(y_pred - y_ref));

fprintf('MAE = %.3e\n',MAE);
fprintf('Max Error = %.3e\n',MaxErr);

figure;
plot(tt,y_ref,'b','LineWidth',2); hold on;
plot(tt,y_pred,'r--','LineWidth',1.8);
legend('dde23 (reference)','PINN');
xlabel('t'); ylabel('y(t)');
title('Pantograph Delay Equation');
grid on;

end

% =====================================================
function [loss,grads] = lossFun(net,t,y0)

y = forward(net,t);
dy = dlgradient(sum(y,'all'),t);

t_delay = 0.5*t;
y_delay = forward(net,t_delay);

res = dy + y - 0.5*y_delay - sin(t);
loss = mean(res.^2) + (y(1)-y0)^2;

grads = dlgradient(loss,net.Learnables);
end

% =====================================================
function dydt = dde_rhs(t,y,Z)
dydt = -y + 0.5*Z + sin(t);
end
