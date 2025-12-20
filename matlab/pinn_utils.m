% =====================================================
% pinn_utils.m
% Utility functions for PINN experiments
% =====================================================

function pinn_utils()
% This file contains helper utilities used across PINN scripts.
% It is not meant to be executed directly.
end

% -----------------------------------------------------
% Mean Absolute Error
% -----------------------------------------------------
function mae = computeMAE(y_pred,y_true)
    y_pred = y_pred(:);
    y_true = y_true(:);
    mae = mean(abs(y_pred - y_true));
end

% -----------------------------------------------------
% Maximum Absolute Error
% -----------------------------------------------------
function maxErr = computeMaxError(y_pred,y_true)
    y_pred = y_pred(:);
    y_true = y_true(:);
    maxErr = max(abs(y_pred - y_true));
end

% -----------------------------------------------------
% Create uniform collocation points
% -----------------------------------------------------
function t = uniformCollocation(T,N)
    t = linspace(0,T,N)';
end

% -----------------------------------------------------
% Convert vectorized NN output to matrix (Riccati)
% -----------------------------------------------------
function X = vec2mat(v,n)
    X = reshape(v,n,n);
end

% -----------------------------------------------------
% Frobenius norm of symmetry error
% -----------------------------------------------------
function err = symmetryError(X)
    err = norm(X - X','fro');
end
