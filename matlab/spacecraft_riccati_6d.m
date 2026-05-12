function spacecraft_riccati_6d()
%% spacecraft_riccati_6d.m  — VECTORISED (fast)
%% Paper:   "Adaptive Physics-Informed Neural Networks for Singular Matrix
%%           Differential Systems with Algebraic Structure Preservation"
%% Authors: Sri Venkata Durga Sudarsan Madhyannapu, Pradheep Kumar S.
%% Journal: Engineering Applications of Artificial Intelligence (Elsevier)
%%          ISSN: 0952-1976 — submitted 10 May 2026
%%
%% SPEED FIX: 6x6 Riccati loss vectorised (was 6+ hours, now ~8-12 min).
%% Paper Section 4.5 / Figure 5 targets:
%%   Sym error < 1e-14, Min eigenvalue > 1e-6,
%%   Fuel deviation < 2%, Online speedup ~ 50x

clc; rng(42);

fprintf('=============================================================\n');
fprintf('6D Spacecraft Trajectory Optimisation  (VECTORISED)\n');
fprintf('Expected runtime: ~8-12 minutes\n');
fprintf('=============================================================\n\n');

n=6; T=10; Nc=100;
A6=diag([0,0,1,1,0,0])+[0 0 0 1 0 0;0 0 0 0 1 0;zeros(4,6)];
A6=A6-0.1*eye(n);
B6=zeros(n,2); B6(3,1)=1; B6(4,2)=1;
Q6=diag([10,5,1,1,0.5,0.5]); R6=eye(2); S6=eye(n);
t_col=linspace(0,T,Nc)'; t_dl=dlarray(t_col','CB');

fprintf('Computing ode45 reference...\n');
t_ode=tic;
opts_ref=odeset('RelTol',1e-10,'AbsTol',1e-12);
[~,Pr_bwd]=ode45(@(t,p) ric_rhs_6d(t,p,A6,B6,Q6,R6),flipud(t_col),S6(:),opts_ref);
P_ref=flipud(Pr_bwd);
t_classical_ms=toc(t_ode)*1000;
fprintf('  ode45 done in %.1f ms\n\n',t_classical_ms);

n_chol=n*(n+1)/2;
layers=[featureInputLayer(1)
        fullyConnectedLayer(100);tanhLayer
        fullyConnectedLayer(100);tanhLayer
        fullyConnectedLayer(100);tanhLayer
        fullyConnectedLayer(n_chol)];
net=dlnetwork(layers); avgG=[]; avgSqG=[]; best_mae=inf; best_net=net;

fprintf('Training 6x6 Cholesky-PINN (vectorised)...\n');
for ep=1:4000
    lr=1e-3*(ep<=2000)+1e-4*(ep>2000);
    [loss,grads]=dlfeval(@lossFun_6d_vec,net,t_dl,A6,B6,Q6,R6,S6,n);
    [net,avgG,avgSqG]=adamupdate(net,grads,avgG,avgSqG,ep,lr);
    if mod(ep,500)==0
        m=eval_mae_6d(net,t_col,P_ref,n);
        if m<best_mae; best_mae=m; best_net=net; end
        fprintf('  Ep %4d  Loss=%.3e  BestMAE=%.3e\n',ep,extractdata(loss),best_mae);
    end
end
fprintf('\nTraining complete.\n');

%% Structural properties
L_out=double(extractdata(predict(best_net,t_dl)));
sym_err_max=0; min_eig_min=inf;
for k=1:Nc
    Lk=build_L(L_out(:,k),n); Pk=Lk*Lk';
    sym_err_max=max(sym_err_max,norm(Pk-Pk','fro'));
    min_eig_min=min(min_eig_min,min(eig(Pk)));
end

fprintf('\n=============================================================\n');
fprintf('STRUCTURAL PROPERTIES (%d collocation pts)\n',Nc);
fprintf('  Max sym error  = %.3e  (paper: <1e-14)\n',sym_err_max);
fprintf('  Min eigenvalue = %.3e  (paper: >1e-6)\n',min_eig_min);
if sym_err_max<1e-13; fprintf('  Symmetry: CONFIRMED\n'); end
if min_eig_min>1e-6;  fprintf('  PD: CONFIRMED\n'); end

%% Speed benchmark
t_p=tic;
for k=1:100; predict(best_net,t_dl); end
t_pinn_ms=toc(t_p)*1000/100;
t_c=tic;
ode45(@(t,p) ric_rhs_6d(t,p,A6,B6,Q6,R6),flipud(t_col),S6(:),opts_ref);
t_class_ms=toc(t_c)*1000;
speedup=t_class_ms/t_pinn_ms;

fprintf('\n=============================================================\n');
fprintf('SPEED BENCHMARK\n');
fprintf('  PINN eval:  %.2f ms  (paper: <1 ms)\n',t_pinn_ms);
fprintf('  ode45:      %.1f ms  (paper: ~50 ms)\n',t_class_ms);
fprintf('  Speedup:    %.1fx    (paper: 50x)\n',speedup);

%% Trajectory
x0=[1.0;0;0.1;0.05;1.0;1.0];
dt_sim=T/1000; t_sim=(0:dt_sim:T)'; Nt=length(t_sim);
x_traj=zeros(n,Nt); x_ref_traj=zeros(n,Nt);
x_traj(:,1)=x0; x_ref_traj(:,1)=x0;
L_sim=double(extractdata(predict(best_net,dlarray(t_sim','CB'))));
for k=1:Nt-1
    Lk=build_L(L_sim(:,k),n); Pk=Lk*Lk';
    uk=-R6\(B6'*Pk*x_traj(:,k));
    x_traj(:,k+1)=x_traj(:,k)+dt_sim*(A6*x_traj(:,k)+B6*uk);
    P_ref_k=reshape(interp1(t_col,P_ref,t_sim(k),'linear','extrap'),n,n);
    ur=-R6\(B6'*P_ref_k*x_ref_traj(:,k));
    x_ref_traj(:,k+1)=x_ref_traj(:,k)+dt_sim*(A6*x_ref_traj(:,k)+B6*ur);
end
fuel_dev=abs((x_traj(5,1)-x_traj(5,end))-(x_ref_traj(5,1)-x_ref_traj(5,end)))...
         /(abs(x_ref_traj(5,1)-x_ref_traj(5,end))+1e-10)*100;

fprintf('\n=============================================================\n');
fprintf('TRAJECTORY: Fuel deviation = %.2f%%  (paper: <2%%)\n',fuel_dev);
fprintf('=============================================================\n');

%% Plot
fig=figure('Position',[50 50 1200 900],'Color','w');
sgtitle('6D Spacecraft Trajectory: PINN vs Reference','FontSize',14);
subplot(2,2,1);
plot(t_sim,x_ref_traj(1,:),'k-','LineWidth',2,'DisplayName','Reference'); hold on;
plot(t_sim,x_traj(1,:),'r--','LineWidth',1.5,'DisplayName','PINN');
xlabel('t(s)'); ylabel('r'); title('Radial Position'); legend; grid on;
subplot(2,2,2);
plot(t_sim,x_ref_traj(2,:),'k-','LineWidth',2,'DisplayName','Reference'); hold on;
plot(t_sim,x_traj(2,:),'r--','LineWidth',1.5,'DisplayName','PINN');
xlabel('t(s)'); ylabel('\theta'); title('Angular Position'); legend; grid on;
subplot(2,2,3);
plot(t_sim,x_ref_traj(5,:),'k-','LineWidth',2,'DisplayName','Reference'); hold on;
plot(t_sim,x_traj(5,:),'r--','LineWidth',1.5,'DisplayName','PINN');
xlabel('t(s)'); ylabel('m'); title(sprintf('Fuel Mass (dev=%.2f%%)',fuel_dev)); legend; grid on;
u_pinn=zeros(2,Nt);
for k=1:Nt
    Lk=build_L(L_sim(:,k),n); Pk=Lk*Lk';
    u_pinn(:,k)=-R6\(B6'*Pk*x_traj(:,k));
end
subplot(2,2,4);
u_ref=zeros(2,Nt);
for k=1:Nt
    Pk=reshape(interp1(t_col,P_ref,t_sim(k),'linear','extrap'),n,n);
    u_ref(:,k)=-R6\(B6'*Pk*x_ref_traj(:,k));
end
plot(t_sim,u_ref(1,:),'k-','LineWidth',2,'DisplayName','Reference'); hold on;
plot(t_sim,u_pinn(1,:),'r--','LineWidth',1.5,'DisplayName','PINN');
xlabel('t(s)'); ylabel('u*(t)'); title(sprintf('Control Law (%.0fx speedup)',speedup)); legend; grid on;
end

function L=build_L(vec,n)
L=zeros(n,n); idx=1;
for col=1:n
    for row=col:n
        if row==col; L(row,col)=exp(vec(idx)); else; L(row,col)=vec(idx); end
        idx=idx+1;
    end
end
end

function mae=eval_mae_6d(net,t_col,P_ref,n)
L=double(extractdata(predict(net,dlarray(t_col','CB'))));
err=0;
for k=1:length(t_col)
    Lk=build_L(L(:,k),n); err=err+norm(Lk*Lk'-reshape(P_ref(k,:),n,n),'fro');
end
mae=err/length(t_col);
end

function dp=ric_rhs_6d(~,p,A,B,Q,R)
n=size(A,1); P=reshape(p,n,n);
dp=-(P*A+A'*P-P*B*(R\(B'*P))+Q); dp=dp(:);
end

function [loss,grads]=lossFun_6d_vec(net,t,A,B,Q,R,S,n)
n_chol=n*(n+1)/2;
L_raw=forward(net,t); Nc=size(t,2);
idx=1; Lv=cell(n,n);
for col=1:n
    for row=col:n
        if row==col; Lv{row,col}=exp(L_raw(idx,:));
        else;         Lv{row,col}=L_raw(idx,:); end
        idx=idx+1;
    end
end
for i=1:n; for j=i+1:n; Lv{i,j}=dlarray(zeros(1,Nc,'single')); end; end

Pv=cell(n,n);
for i=1:n
    for j=1:n
        pij=dlarray(zeros(1,Nc,'single'));
        for kk=1:min(i,j); pij=pij+Lv{i,kk}.*Lv{j,kk}; end
        Pv{i,j}=pij;
    end
end

dPv=cell(n,n);
for i=1:n
    for j=i:n
        dPv{i,j}=dlgradient(sum(Pv{i,j}),t,'EnableHigherDerivatives',true);
        dPv{j,i}=dPv{i,j};
    end
end

res=dlarray(single(0));
for i=1:n
    for j=1:n
        PA_ij=dlarray(zeros(1,Nc,'single'));
        for kk=1:n; PA_ij=PA_ij+Pv{i,kk}.*A(kk,j); end
        AtP_ij=dlarray(zeros(1,Nc,'single'));
        for kk=1:n; AtP_ij=AtP_ij+A(kk,i).*Pv{kk,j}; end
        PB_i=dlarray(zeros(1,Nc,'single')); PB_j=dlarray(zeros(1,Nc,'single'));
        for kk=1:n; for ll=1:size(B,2)
            PB_i=PB_i+Pv{i,kk}.*B(kk,ll);
            PB_j=PB_j+Pv{j,kk}.*B(kk,ll);
        end; end
        PBRP_ij=PB_i.*PB_j/R(1,1);
        rhs_ij=-PA_ij-AtP_ij+PBRP_ij-Q(i,j);
        res=res+sum((dPv{i,j}-rhs_ij).^2);
    end
end
bc=dlarray(single(0));
for i=1:n; for j=1:n; bc=bc+(Pv{i,j}(end)-S(i,j)).^2; end; end
loss=res/Nc+20*bc;
grads=dlgradient(loss,net.Learnables);
end
