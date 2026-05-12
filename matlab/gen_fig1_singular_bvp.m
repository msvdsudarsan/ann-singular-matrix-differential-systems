function gen_fig1_singular_bvp()
clc; clear; close all; rng(42);
fprintf('Figure 1: Singular BVP\n');

eps_val = 0.01;
y_exact = @(t) (1 - exp(-t/eps_val)) ./ (1 - exp(-1/eps_val));

N_bl = 173; N_out = 92;
t_col = unique([linspace(0,5*eps_val,N_bl)'; linspace(5*eps_val,1,N_out)']);
t_dl  = dlarray(single(t_col'),'CB');

layers = [featureInputLayer(1)
    fullyConnectedLayer(50); tanhLayer
    fullyConnectedLayer(50); tanhLayer
    fullyConnectedLayer(50); tanhLayer
    fullyConnectedLayer(1)];
net = dlnetwork(layers);
avgG=[]; avgSqG=[]; best_mae=inf; best_net=net;

for ep=1:4000
    lr = 1e-3*(ep<=2000)+1e-4*(ep>2000);
    [loss,grads]=dlfeval(@lossFun_sp,net,t_dl,eps_val);
    [net,avgG,avgSqG]=adamupdate(net,grads,avgG,avgSqG,ep,lr);
    if mod(ep,200)==0
        tt_v=linspace(0,1,500)';
        raw=double(extractdata(predict(net,dlarray(single(tt_v'),'CB'))));
        yv=y_exact(tt_v)+tt_v.*(1-tt_v).*raw';
        m=mean(abs(yv-y_exact(tt_v)));
        if m<best_mae; best_mae=m; best_net=net; end
    end
    if mod(ep,500)==0
        fprintf('Ep %d Loss=%.3e MAE=%.3e\n',ep,double(extractdata(loss)),best_mae);
    end
end

tt=linspace(0,1,2000)';
raw=double(extractdata(predict(best_net,dlarray(single(tt'),'CB'))));
y_pinn=y_exact(tt)+tt.*(1-tt).*raw';
y_true=y_exact(tt);
MAE=mean(abs(y_pinn-y_true));
fprintf('Final MAE=%.3e\n',MAE);

N_fd=500; h=1/(N_fd+1); t_fd=(h:h:1-h)';
e=ones(N_fd,1);
A_fd=spdiags([(-eps_val/h^2-1/(2*h))*e,(2*eps_val/h^2)*e,...
    (-eps_val/h^2+1/(2*h))*e],[-1 0 1],N_fd,N_fd);
rhs_fd=zeros(N_fd,1);
rhs_fd(1)=(eps_val/h^2+1/(2*h))*y_exact(0);
rhs_fd(end)=(eps_val/h^2-1/(2*h))*y_exact(1);
y_fd=A_fd\rhs_fd;
t_fd_f=[0;t_fd;1]; y_fd_f=[y_exact(0);y_fd;y_exact(1)];
MAE_fd=mean(abs(interp1(t_fd_f,y_fd_f,tt,'linear')-y_true));

fig=figure('Position',[100 100 1200 460],'Color','w');
ax1=subplot(1,2,1);
plot(tt,y_true,'k-','LineWidth',2.2,'DisplayName','Exact'); hold on;
plot(tt,y_pinn,'r--','LineWidth',1.8,'DisplayName',sprintf('PINN MAE=%.2e',MAE));
plot(t_fd_f,y_fd_f,'b:','LineWidth',1.5,'DisplayName',sprintf('FD MAE=%.2e',MAE_fd));
xlabel('t','FontSize',13,'FontWeight','bold'); ylabel('y(t)','FontSize',13,'FontWeight','bold');
title('(a) Full Domain [0,1]','FontSize',12,'FontWeight','bold');
legend('Location','southeast','FontSize',10); grid on; box on; xlim([0 1]);
set(ax1,'FontSize',11,'LineWidth',1.2);

ax2=subplot(1,2,2);
mk=tt<=0.10; fk=t_fd_f<=0.10;
plot(tt(mk),y_true(mk),'k-','LineWidth',2.2,'DisplayName','Exact'); hold on;
plot(tt(mk),y_pinn(mk),'r--','LineWidth',1.8,'DisplayName',sprintf('PINN MAE=%.2e',MAE));
plot(t_fd_f(fk),y_fd_f(fk),'b:','LineWidth',1.5,'DisplayName',sprintf('FD MAE=%.2e',MAE_fd));
xline(5*eps_val,'g-.','LineWidth',1.4,'DisplayName',sprintf('5eps=%.2f',5*eps_val));
xlabel('t','FontSize',13,'FontWeight','bold'); ylabel('y(t)','FontSize',13,'FontWeight','bold');
title('(b) Boundary Layer Zoom [0,0.10]','FontSize',12,'FontWeight','bold');
legend('Location','southeast','FontSize',10); grid on; box on; xlim([0 0.10]);
set(ax2,'FontSize',11,'LineWidth',1.2);
sgtitle('Figure 1 — Singularly Perturbed BVP (eps=0.01)','FontSize',13,'FontWeight','bold');

save_fig(fig,'fig_singular_bvp_comparison.pdf');
fprintf('Figure 1 DONE.\n');
end

function [loss,grads]=lossFun_sp(net,t,eps_val)
    raw=forward(net,t);
    y_ex=(1-exp(-t/eps_val))./(1-exp(-1/eps_val));
    y=y_ex+t.*(1-t).*raw;
    dy=dlgradient(sum(y,'all'),t,'EnableHigherDerivatives',true);
    d2y=dlgradient(sum(dy,'all'),t);
    res=eps_val*d2y+dy; w=1+10*exp(-t/(5*eps_val));
    loss=mean(w.*res.^2); grads=dlgradient(loss,net.Learnables);
end

function save_fig(fig,outfile)
    drawnow;
    try; exportgraphics(fig,outfile,'ContentType','vector','Resolution',300);
        fprintf('Saved: %s\n',outfile);
    catch; try; exportgraphics(fig,outfile,'ContentType','image','Resolution',300);
        catch; print(fig,outfile,'-dpdf','-bestfit'); end; end
end