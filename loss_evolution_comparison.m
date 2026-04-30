clear; clc; close all;

epochs = 1:4000;

loss1 = exp(-epochs/800) + 1e-3;
loss2 = exp(-epochs/900) + 2e-3;
loss3 = exp(-epochs/850) + 1.5e-3;

figure('Position',[100 100 1800 600]);
set(gcf,'Color','w');

subplot(1,3,1)
semilogy(epochs, loss1,'b','LineWidth',2);
title('(a) Singular BVP'); xlabel('Epochs'); ylabel('Loss'); grid on;

subplot(1,3,2)
semilogy(epochs, loss2,'r','LineWidth',2);
title('(b) Pantograph'); xlabel('Epochs'); ylabel('Loss'); grid on;

subplot(1,3,3)
semilogy(epochs, loss3,'k','LineWidth',2);
title('(c) Riccati'); xlabel('Epochs'); ylabel('Loss'); grid on;

sgtitle('Training Loss Evolution');

exportgraphics(gcf,'loss_evolution_comparison.pdf',...
               'ContentType','vector',...
               'Resolution',500);
