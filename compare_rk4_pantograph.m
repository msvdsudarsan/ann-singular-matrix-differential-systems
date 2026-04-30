% compare_rk4_pantograph.m
% RK4 + interpolation baseline for pantograph equation
% y'(t) = -y(t) + y(t/2), y(0)=1

clear; clc;

T = 1;
N = 200;
t = linspace(0,T,N+1)';
h = t(2)-t(1);

% Exact solution (reference)
y_exact_fun = @(t) exp(-t) .* (1 + t);

y = zeros(N+1,1);
y(1) = 1;

for i = 1:N
    ti = t(i);

    if i == 1
        % no interpolation possible yet
        f = @(tt,yy) -yy + 1;
    else
        f = @(tt,yy) -yy + interp1(t(1:i), y(1:i), tt/2, 'linear','extrap');
    end

    k1 = f(ti, y(i));
    k2 = f(ti + h/2, y(i) + h*k1/2);
    k3 = f(ti + h/2, y(i) + h*k2/2);
    k4 = f(ti + h,   y(i) + h*k3);

    y(i+1) = y(i) + h*(k1 + 2*k2 + 2*k3 + k4)/6;
end

% Errors
y_exact = y_exact_fun(t);
abs_err = abs(y - y_exact);

MAE = mean(abs_err);
MaxErr = max(abs_err);

fprintf('RK4 MAE = %.3e\n', MAE);
fprintf('RK4 Max Error = %.3e\n', MaxErr);
