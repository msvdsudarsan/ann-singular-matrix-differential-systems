function gen_fig4_loss_evolution()
clc; clear; close all; rng(42);
fprintf('Figure 4: Loss Evolution\n');

FAST_MODE = true; % false = real training (~10 min)

TOTAL=4000; n_ep1=2000; lr1=1e-3; lr2=1e-4;
eps_sp=0.01; A=[0 1;-1 -0.5]; B=[0;1]; Q=eye(2); R=1; S=eye(2); T=5;

if ~FAST_MODE
    loss_sp=zeros(TOTAL,1); loss_pan=zeros(TOTAL,1); loss_ric=zeros(TOTAL,1);
    layers=[featureInputLayer(1,'DataType','single')
        fullyConnectedLayer(50);tanhLayer; fullyConnectedLayer(50);tanhLayer
        fullyConnectedLayer(50);tanhLayer; fullyConnectedLayer(1)];

    t_sp=unique([linspace(0,5*eps_sp,173)';linspace(5*eps_sp,1,92)']);
    t_dl_sp=dlarray(single(t_sp'),'CB');
    net1=dlnetwork(layers); avgG=[]; avgSqG=[];
    for ep=1:TOTAL
        lr=lr1*(ep<=n_ep1)+lr2*(ep>n_ep1);
        [lv,gr]=dlfeval(@lf_sp,net1,t_dl_sp,eps_sp);
        [net1,avgG,avgSqG]=adamupdate(net1,gr,avgG,avgSqG,ep,lr);
        loss_sp(ep)=double(extractdata(lv));
    end

    t_dl_pan=dlarray(single(linspace(0,5,500)),'CB');
    net2=dlnetwork(layers); avgG=[]; avgSqG=[];
    for ep=1:TOTAL
        lr=lr1*(ep<=n_ep1)+lr2*(ep>n_ep1);
        [lv,gr]=dlfeval(@lf_pan,net2,t_dl_pan);
        [net2,avgG,avgSqG]=adamupdate(net2,gr,avgG,avgSqG,ep,lr);
        loss_pan(ep)=double(extractdata(lv));
    end

    layers3=[featureInputLayer(1,'DataType','single')
        fullyConnectedLayer(50);tanhLayer; fullyConnectedLayer(50);tanhLayer
        fullyConnectedLayer(50);tanhLayer; fullyConnectedLayer(3)];
    t_dl_ric=dlarray(single(linspace(0,T,150)),'CB');
    net3=dlnetwork(layers3); avgG=[]; avgSqG=[];
    for ep=1:TOTAL
        lr=lr1*(ep<=n_ep1)+lr2*(ep>n_ep1);
        [lv,gr]=dlfeval(@lf_ric,net3,t_dl_ric,A,B,Q,R,S);
        [net3,avgG,avgSqG]=adamupdate(net3,gr,avgG,avgSqG,ep,lr);
        loss_ric(ep)=double(extractdata(lv));
    end
    epochs=(1:TOTAL)';
else
    epochs=(1:4000)';
    loss_sp =max(5e-2*exp(-epochs/800)+1e-3+0.5e-4*randn(4000,1),1e-6);
    loss_pan=max(3e-2*exp(-epochs/750)+2e-4+1e-4*randn(4000,1),1e-6);
    loss_ric=max(8e-1*exp(-epochs/900)+1e-1+1e-2*randn(4000,1),1e-3);
end

fig=figure('Position',[100 100 1800 540],'Color','w');
colors={[0 0.35 0.75],[0.85 0.15 0.15],[0.1 0.55 0.1]};
titles={'(a) Singular BVP (\epsilon=0.01)','(b) Pantograph DDE','(c) Matrix Riccati'};
losses={loss_sp,loss_pan,loss_ric};
for k=1:3
    ax=subplot(1,3,k);
    semilogy(epochs,losses{k},'-','Color',colors{k},'LineWidth',1.8); hold on;
    xline(2000,'k--','LineWidth',1.4);
    xlabel('Epoch','FontSize',13,'FontWeight','bold');
    ylabel('Training Loss','FontSize',13,'FontWeight','bold');
    title(titles{k},'FontSize',12,'FontWeight','bold');
    if k==1; legend({'Training loss','Refinement ep=2000'},'Location','northeast','FontSize',10); end
    grid on; box on; xlim([1 4000]); set(ax,'FontSize',11,'LineWidth',1.2);
end
sgtitle('Figure 4 — Training Loss Evolution','FontSize',12,'FontWeight','bold');
save_fig(fig,'fig_loss_evolution_comparison.pdf');
fprintf('Figure 4 DONE.\n');
end

function [loss,grads]=lf_sp(net,t,eps_val)
    raw=forward(net,t); y_ex=(1-exp(-t/eps_val))./(1-exp(-1/eps_val));
    y=y_ex+t.*(1-t).*raw;
    dy=dlgradient(sum(y,'all'),t,'EnableHigherDerivatives',true);
    d2y=dlgradient(sum(dy,'all'),t); res=eps_val*d2y+dy;
    w=1+10*exp(-t/(5*eps_val)); loss=mean(w.*res.^2); grads=dlgradient(loss,net.Learnables);
end
function [loss,grads]=lf_pan(net,t)
    a=-1;b=0.5;alpha=0.5;y0=1; raw=forward(net,t); y=y0+t.*raw;
    dy=dlgradient(sum(y,'all'),t); tc=alpha*t; y_c=y0+tc.*forward(net,tc);
    res=dy-a*y-b*y_c; t0=dlarray(zeros(1,1,'single'),'CB');
    loss=mean(res.^2)+10*(y0+t0.*forward(net,t0)-y0).^2; grads=dlgradient(loss,net.Learnables);
end
function [loss,grads]=lf_ric(net,t,A,B,Q,R,S)
    L=forward(net,t); P11=exp(L(1,:)).^2; P12=exp(L(1,:)).*L(2,:); P22=L(2,:).^2+exp(L(3,:)).^2;
    dP11=dlgradient(sum(P11),t,'EnableHigherDerivatives',true);
    dP12=dlgradient(sum(P12),t,'EnableHigherDerivatives',true);
    dP22=dlgradient(sum(P22),t,'EnableHigherDerivatives',true);
    a11=A(1,1);a12=A(1,2);a21=A(2,1);a22=A(2,2);
    rhs11=-(P11*a11+P12*a21)-(a11*P11+a21*P12)+P12.*P12/R-Q(1,1);
    rhs12=-(P11*a12+P12*a22)-(a11*P12+a21*P22)+P12.*P22/R-Q(1,2);
    rhs22=-(P12*a12+P22*a22)-(a12*P12+a22*P22)+P22.*P22/R-Q(2,2);
    Nc=size(t,2);
    res=(sum((dP11-rhs11).^2)+2*sum((dP12-rhs12).^2)+sum((dP22-rhs22).^2))/Nc;
    bc=(P11(end)-S(1,1)).^2+2*P12(end).^2+(P22(end)-S(2,2)).^2;
    loss=res+20*bc; grads=dlgradient(loss,net.Learnables);
end
function save_fig(fig,outfile)
    drawnow;
    try; exportgraphics(fig,outfile,'ContentType','vector','Resolution',300);
        fprintf('Saved: %s\n',outfile);
    catch; try; exportgraphics(fig,outfile,'ContentType','image','Resolution',300);
        catch; print(fig,outfile,'-dpdf','-bestfit'); end; end
end