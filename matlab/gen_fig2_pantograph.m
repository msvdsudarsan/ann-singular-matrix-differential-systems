function gen_fig2_pantograph()
% =========================================================================
% gen_fig2_pantograph.m  --  EAAI Paper Figure 2
% =========================================================================
% Pantograph-type delay differential equation:
%   y'(t) = -y(t) + 0.5*y(0.5*t),   y(0) = 1,   t in [0, 5]
% Trial function:  y_theta(t) = 1 + t * N(t)  => y(0)=1 exactly
% Network:  1 -> 50 -> 50 -> 1  (tanh hidden, linear output)
% Optimizer: Adam, lr=1e-3, 6000 epochs
% =========================================================================
rng(42);
T_end  = 5;   Nc = 200;   epochs = 6000;   lr = 1e-3;   alpha = 0.5;

%% Network (Xavier init)
W1=randn(50,1 )*sqrt(2/51);   b1=zeros(50,1);
W2=randn(50,50)*sqrt(2/100);  b2=zeros(50,1);
W3=randn(1, 50)*sqrt(2/51);   b3=zeros(1,1);

%% Adam states
[mW1,vW1]=deal(zeros(size(W1))); [mb1,vb1]=deal(zeros(size(b1)));
[mW2,vW2]=deal(zeros(size(W2))); [mb2,vb2]=deal(zeros(size(b2)));
[mW3,vW3]=deal(zeros(size(W3))); [mb3,vb3]=deal(zeros(size(b3)));

t_col  = linspace(0.001, T_end, Nc)';
loss_h = zeros(epochs,1);

fprintf('Training pantograph PINN  (1->50->50->1, 6000 epochs)...\n');

for ep = 1:epochs
    t  = t_col(:)';
    ta = alpha * t;

    % Forward at t
    z1=W1*t+b1;   a1=tanh(z1);
    z2=W2*a1+b2;  a2=tanh(z2);
    Nt=W3*a2+b3;
    y_t=(1+t.*Nt)';

    % dy/dt via complex step
    h_cs=1e-20; tc=t+1i*h_cs;
    a1c=tanh(W1*tc+b1); a2c=tanh(W2*a1c+b2); Ntc=W3*a2c+b3;
    dy_t=(imag(1+tc.*Ntc)/h_cs)';

    % Forward at alpha*t
    a1a=tanh(W1*ta+b1); a2a=tanh(W2*a1a+b2); Nta=W3*a2a+b3;
    y_at=(1+ta.*Nta)';

    % Residual
    res  = dy_t - (-y_t + 0.5*y_at);
    loss = mean(res.^2);
    loss_h(ep) = loss;

    % Backprop
    dr    = (2/Nc)*res(:)';
    dL_dN = dr.*t + dr;          % y_t term + dy/dt term

    gW3=dL_dN*a2';  gb3=sum(dL_dN,2);
    dL_da2=W3'*dL_dN; dL_dz2=dL_da2.*(1-a2.^2);
    gW2=dL_dz2*a1'; gb2=sum(dL_dz2,2);
    dL_da1=W2'*dL_dz2; dL_dz1=dL_da1.*(1-a1.^2);
    gW1=dL_dz1*t';  gb1=sum(dL_dz1,2);

    % Adam
    b1_=0.9; b2_=0.999; ep_=1e-8;
    [W1,mW1,vW1]=adam(W1,gW1,mW1,vW1,lr,b1_,b2_,ep_,ep);
    [b1,mb1,vb1]=adam(b1,gb1,mb1,vb1,lr,b1_,b2_,ep_,ep);
    [W2,mW2,vW2]=adam(W2,gW2,mW2,vW2,lr,b1_,b2_,ep_,ep);
    [b2,mb2,vb2]=adam(b2,gb2,mb2,vb2,lr,b1_,b2_,ep_,ep);
    [W3,mW3,vW3]=adam(W3,gW3,mW3,vW3,lr,b1_,b2_,ep_,ep);
    [b3,mb3,vb3]=adam(b3,gb3,mb3,vb3,lr,b1_,b2_,ep_,ep);

    if mod(ep,500)==0
        fprintf('  Epoch %5d | Loss = %.4e\n', ep, loss);
    end
end

%% Reference RK4
fprintf('\nBuilding RK4 reference (N=6000)...\n');
t_ref = linspace(0,T_end,6001);
y_ref = rk4_panto(t_ref);

%% Evaluate PINN
te=t_ref(:)';
z1e=W1*te+b1; a1e=tanh(z1e);
z2e=W2*a1e+b2; a2e=tanh(z2e);
Nte=W3*a2e+b3;
y_pinn=(1+te.*Nte)';

%% dde23
opts_d=ddeset('RelTol',1e-8,'AbsTol',1e-10);
sol=dde23(@(t,y,Z)-y+0.5*Z, 0.5, @(t)1, [0 T_end], opts_d);
y_dde=deval(sol,t_ref)';

%% Metrics
yr=y_ref(:);
mae_p=mean(abs(y_pinn-yr)); max_p=max(abs(y_pinn-yr));
mae_d=mean(abs(y_dde-yr));  max_d=max(abs(y_dde-yr));
fprintf('\n============================================\n');
fprintf('PINN   MAE=%.4e  MaxErr=%.4e\n',mae_p,max_p);
fprintf('dde23  MAE=%.4e  MaxErr=%.4e\n',mae_d,max_d);
fprintf('Paper: MAE~9.27e-4,  MaxErr~1.83e-3\n');
fprintf('============================================\n');

%% Figure 2
fig=figure('Position',[100 100 800 520],'Color','w');
plot(t_ref,yr,    'k-', 'LineWidth',2.2,'DisplayName','Reference (RK4,N=6000)');
hold on;
plot(t_ref,y_pinn,'r--','LineWidth',1.8,'DisplayName',...
    sprintf('PINN (MAE=%.2e, Max=%.2e)',mae_p,max_p));
plot(t_ref,y_dde, 'b:', 'LineWidth',1.8,'DisplayName',...
    sprintf('dde23 (MAE=%.2e)',mae_d));
hold off;
xlabel('t','FontSize',13); ylabel('y(t)','FontSize',13);
title('Pantograph-Type Delay Differential Equation  [y''(t)=-y(t)+0.5y(0.5t)]','FontSize',13);
legend('Location','southwest','FontSize',10);
grid on; box on; set(gca,'FontSize',12);
save_fig(fig,'fig_pantograph_pinn');
fprintf('Figure 2 saved.\n');
end

%% ── HELPERS ─────────────────────────────────────────────────────────────
function [p,m,v]=adam(p,g,m,v,lr,b1,b2,ep,t)
    m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g.^2;
    p=p-lr*(m/(1-b1^t))./(sqrt(v/(1-b2^t))+ep);
end

function y=rk4_panto(tv)
    N=numel(tv)-1; y=zeros(1,N+1); y(1)=1;
    for i=1:N
        t0=tv(i); h=tv(i+1)-t0; y0=y(i);
        yh=@(s)interp_y(tv(1:i),y(1:i),s);
        k1=h*(-y0        +0.5*yh(0.5*t0));
        k2=h*(-(y0+k1/2) +0.5*yh(0.5*(t0+h/2)));
        k3=h*(-(y0+k2/2) +0.5*yh(0.5*(t0+h/2)));
        k4=h*(-(y0+k3)   +0.5*yh(0.5*(t0+h)));
        y(i+1)=y0+(k1+2*k2+2*k3+k4)/6;
    end
end

function yi=interp_y(tk,yk,tq)
    if tq<=tk(1),yi=yk(1);return;end
    if tq>=tk(end),yi=yk(end);return;end
    yi=interp1(tk,yk,tq,'pchip');
end

function save_fig(fig,fname)
    try;exportgraphics(fig,[fname '.pdf'],'ContentType','image','Resolution',300);
    catch;try;saveas(fig,[fname '.pdf']);catch;print(fig,'-dpdf',fname);end;end
    try;exportgraphics(fig,[fname '.png'],'Resolution',300);
    catch;saveas(fig,[fname '.png']);end
end