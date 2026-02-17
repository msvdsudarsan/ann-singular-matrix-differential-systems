% Pantograph DDE - dde23
sol = dde23(@(t,y,Z) -y + 0.5*Z(:,1), 0.5, 1, [0,1]);

% Reference (high-res RK4)
dt = 1e-4;
t_ref = 0:dt:1;
y_ref = zeros(size(t_ref));
y_ref(1) = 1;
for i = 2:length(t_ref)
    ti = t_ref(i-1);
    idx = max(1, round(0.5*ti/dt)+1);
    k1 = -y_ref(i-1) + 0.5*y_ref(idx);
    y_ref(i) = y_ref(i-1) + dt*k1;
end

% MAE calculation
y_dde23 = deval(sol, t_ref);
MAE_dde23 = mean(abs(y_dde23 - y_ref));
MaxE_dde23 = max(abs(y_dde23 - y_ref));

fprintf('dde23 MAE = %.4e\n', MAE_dde23);
fprintf('dde23 Max Error = %.4e\n', MaxE_dde23);
