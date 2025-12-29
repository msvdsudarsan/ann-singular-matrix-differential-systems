function bvp4c_singular_test()

eps = 0.01;

% Exact solution
y_exact = @(t) (1-exp(-t/eps))./(1-exp(-1/eps));

% ODE system: y1 = y, y2 = y'
odefun = @(t,y) [ y(2);
                 -y(2)/eps ];

% Boundary conditions
bcfun = @(ya,yb) [ ya(1);      % y(0)=0
                   yb(1)-1 ];  % y(1)=1

% Initial mesh and guess
t_init = linspace(0,1,10);
solinit = bvpinit(t_init,[0;0]);

% Solve
sol = bvp4c(odefun,bcfun,solinit);

% Evaluation grid (same as PINN)
t_eval = linspace(0,1,1000);
y_bvp = deval(sol,t_eval);
y_bvp = y_bvp(1,:);

% Exact
y_true = y_exact(t_eval);

% Errors
MAE = mean(abs(y_bvp - y_true));
MaxErr = max(abs(y_bvp - y_true));

fprintf('bvp4c MAE = %.3e\n', MAE);
fprintf('bvp4c Max Error = %.3e\n', MaxErr);

end
