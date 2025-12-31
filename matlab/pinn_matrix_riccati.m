function pinn_matrix_riccati()
clc; clear; close all;

numRuns = 3;
MAE_all = zeros(numRuns,1);

for seed = 1:numRuns
    rng(seed);
    fprintf('\n--- Riccati PINN | Seed %d ---\n',seed);
    tic;
    
    % System matrices
    A = [0 1; -1 -0.5];
    B = [0; 1];
    Q = eye(2);
    R = 1;
    S = eye(2);
    T = 5;
    
    % Collocation points
    Nc = 150;
    t = linspace(0,T,Nc)';
    
    % Network initialization
    layers = [
        featureInputLayer(1)
        fullyConnectedLayer(48)
        tanhLayer
        fullyConnectedLayer(48)
        tanhLayer
        fullyConnectedLayer(3)  % Output: [l11, l21, l22]
    ];
    net = dlnetwork(layers);
    
    % Training parameters
    lr = 1e-3;
    epochs = 4000;
    avgGrad = [];
    avgSqGrad = [];
    
    % Training loop
    for epoch = 1:epochs
        [loss,grads] = dlfeval(@lossFun,net,t,A,B,Q,R,S,T);
        [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,lr);
        
        if mod(epoch,500)==0
            fprintf('Seed %d | Epoch %4d | Loss = %.3e\n', ...
                seed, epoch, extractdata(loss));
        end
    end
    
    fprintf('Training done (Seed %d). Time = %.1f s\n',seed,toc);
    
    % ===== EVALUATION =====
    tt = linspace(0,T,200)';
    
    % PINN solution
    P_pinn = zeros(2,2,length(tt));
    for k = 1:length(tt)
        t_k = dlarray(tt(k),'CB');
        L = extractdata(predict(net,t_k));
        l11 = L(1); l21 = L(2); l22 = L(3);
        
        P11 = l11^2 + 1e-3;
        P12 = l11 * l21;
        P22 = l21^2 + l22^2 + 1e-3;
        
        P_pinn(:,:,k) = [P11 P12; P12 P22];
    end
    
    % Reference solution (ode45)
    [~,P_ref_vec] = ode45(@(t,p) riccati_rhs(t,p,A,B,Q,R), ...
                          flipud(tt), S(:));
    P_ref_vec = flipud(P_ref_vec);
    P_ref = zeros(2,2,length(tt));
    for k = 1:length(tt)
        P_ref(:,:,k) = reshape(P_ref_vec(k,:),2,2);
    end
    
    % Compute MAE
    err = 0;
    for k = 1:length(tt)
        err = err + norm(P_pinn(:,:,k) - P_ref(:,:,k),'fro');
    end
    MAE_all(seed) = err / length(tt);
    
    % ===== GENERATE FIGURE (LAST SEED ONLY) =====
    if seed == numRuns
        % Compute traces
        trace_pinn = zeros(length(tt),1);
        trace_ref  = zeros(length(tt),1);
        for k = 1:length(tt)
            trace_pinn(k) = trace(P_pinn(:,:,k));
            trace_ref(k)  = trace(P_ref(:,:,k));
        end
        
        % Create figure with explicit settings
        fig = figure('Position', [100 100 800 600], ...
                     'Color', 'w', ...
                     'Renderer', 'painters');
        
        % Plot
        plot(tt, trace_ref, 'k-', 'LineWidth', 2.5, 'DisplayName', 'ode45 (reference)'); 
        hold on;
        plot(tt, trace_pinn, 'r--', 'LineWidth', 2.0, 'DisplayName', 'PINN (structure-preserving)');
        
        % Formatting
        xlabel('Time t', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('trace(P(t))', 'FontSize', 12, 'FontWeight', 'bold');
        title('Matrix Riccati Equation: Trace Evolution', 'FontSize', 14);
        legend('Location', 'best', 'FontSize', 11);
        grid on;
        box on;
        set(gca, 'FontSize', 11, 'LineWidth', 1.2);
        
        % Force rendering
        drawnow;
        pause(0.5);
        
        % Save figure
        saveas(fig, 'figure_riccati_trace.png');
        exportgraphics(fig, 'figure_riccati_trace.pdf', 'ContentType', 'vector');
        
        fprintf('Figure saved: figure_riccati_trace.png and .pdf\n');
    end
end

% Print final statistics
fprintf('\n========================================\n');
fprintf('Riccati MAE = %.3e ± %.3e\n', mean(MAE_all), std(MAE_all));
fprintf('========================================\n');

end

%% ===== LOSS FUNCTION WITH FINITE DIFFERENCES =====
function [loss,grads] = lossFun(net,t,A,B,Q,R,S,T)
    % Small perturbation for finite differences
    dt = 1e-4;
    
    Nc = length(t);
    res = 0;
    
    for k = 1:Nc
        t_k = dlarray(t(k),'CB');
        
        % Current point
        L = forward(net,t_k);
        l11 = L(1); l21 = L(2); l22 = L(3);
        P11 = l11^2 + 1e-3;
        P12 = l11 * l21;
        P22 = l21^2 + l22^2 + 1e-3;
        P = [P11 P12; P12 P22];
        
        % Time derivative via finite differences
        if k < Nc
            t_next = dlarray(t(k)+dt,'CB');
            L_next = forward(net,t_next);
            l11_n = L_next(1); l21_n = L_next(2); l22_n = L_next(3);
            P11_n = l11_n^2 + 1e-3;
            P12_n = l11_n * l21_n;
            P22_n = l21_n^2 + l22_n^2 + 1e-3;
            P_next = [P11_n P12_n; P12_n P22_n];
            
            dPdt = (P_next - P) / dt;
        else
            % Use backward difference at final point
            t_prev = dlarray(t(k)-dt,'CB');
            L_prev = forward(net,t_prev);
            l11_p = L_prev(1); l21_p = L_prev(2); l22_p = L_prev(3);
            P11_p = l11_p^2 + 1e-3;
            P12_p = l11_p * l21_p;
            P22_p = l21_p^2 + l22_p^2 + 1e-3;
            P_prev = [P11_p P12_p; P12_p P22_p];
            
            dPdt = (P - P_prev) / dt;
        end
        
        % Riccati residual
        ric = -P*A - A'*P + P*B*(B'*P) - Q;
        res = res + sum((dPdt - ric).^2,'all');
    end
    
    % Terminal condition at t=T
    t_T = dlarray(T,'CB');
    L_T = forward(net,t_T);
    l11_T = L_T(1); l21_T = L_T(2); l22_T = L_T(3);
    P11_T = l11_T^2 + 1e-3;
    P12_T = l11_T * l21_T;
    P22_T = l21_T^2 + l22_T^2 + 1e-3;
    PT = [P11_T P12_T; P12_T P22_T];
    
    loss_tc = sum((PT - S).^2,'all');
    loss = res/Nc + 10*loss_tc;
    
    grads = dlgradient(loss,net.Learnables);
end

%% ===== RICCATI ODE RHS =====
function dp = riccati_rhs(~,p,A,B,Q,R)
    P = reshape(p,2,2);
    dP = -P*A - A'*P + P*B*(B'*P) - Q;
    dp = dP(:);
end
