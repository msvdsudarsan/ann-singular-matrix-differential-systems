% compare_finite_difference.m
% Finite Difference baseline for singularly perturbed BVP
% Problem: eps*y'' + y' = 0, y(0)=0, y(1)=1

clear; clc;

% Parameters
eps = 0.01;
N   = 100;                 % uniform grid (same as paper baseline)
t   = linspace(0,1,N+1)';
h   = t(2) - t(1);

% Exact solution
y_exact = (1 - exp(-t/eps)) / (1 - exp(-1/eps));

% Finite difference matrix
A = zeros(N-1,N-1);
b = zeros(N-1,1);

for i = 1:N-1
    % y''
    A(i,i) = -2*eps/h^2;
    if i > 1
        A(i,i-1) = eps/h^2 - 1/(2*h);
    end
    if i < N-1
        A(i,i+1) = eps/h^2 + 1/(2*h);
    end
end

% Boundary conditions
b(1)   = b(1)   - (eps/h^2 - 1/(2*h))*0;   % y(0)=0
b(end) = b(end) - (eps/h^2 + 1/(2*h))*1;   % y(1)=1

% Solve system
y_inner = A \ b;
y_fd = [0; y_inner; 1];

% Errors
abs_err = abs(y_fd - y_exact);
MAE = mean(abs_err);
MaxErr = max(abs_err);

% Print results
fprintf('FD MAE = %.3e\n', MAE);
fprintf('FD Max Error = %.3e\n', MaxErr);
