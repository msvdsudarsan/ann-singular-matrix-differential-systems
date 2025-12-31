function run_all_experiments()
% =====================================================
% Run all PINN experiments AND save figures
% =====================================================
clc; close all;

outdir = 'figures_output';
if ~exist(outdir,'dir')
    mkdir(outdir);
end

fprintf('=================================================\n');
fprintf('PINN Framework for Singular Matrix Systems\n');
fprintf('Reproducing Results from PDF Manuscript\n');
fprintf('=================================================\n\n');

%% ---------------- Problem 1 ----------------
fprintf('>>> Running Problem 1: Singularly Perturbed BVP <<<\n');
fprintf('=================================================\n');
pinn_singular_perturbation();
save_all_figs(outdir,'bvp');

%% ---------------- Problem 2 ----------------
fprintf('\n>>> Running Problem 2: Pantograph Delay Equation <<<\n');
fprintf('=================================================\n');
pinn_pantograph_delay();
save_all_figs(outdir,'pantograph');

%% ---------------- Problem 3 ----------------
fprintf('\n>>> Running Problem 3: Matrix Riccati Equation <<<\n');
fprintf('=================================================\n');
pinn_matrix_riccati();
save_all_figs(outdir,'riccati');

fprintf('\n=================================================\n');
fprintf('ALL EXPERIMENTS COMPLETED SUCCESSFULLY\n');
fprintf('Figures saved in folder: %s\n', outdir);
fprintf('=================================================\n');
end

% =====================================================
function save_all_figs(outdir,prefix)
    figs = findall(0,'Type','figure');
    for k = 1:length(figs)
        % FIXED: Use datetime instead of datestr
        timestamp = string(datetime('now','Format','yyyyMMdd'));
        fname = sprintf('%s_%s_fig%d', timestamp, prefix, k);
        
        % Save as PDF (vector) and PNG (raster)
        saveas(figs(k), fullfile(outdir,[fname '.pdf']));
        saveas(figs(k), fullfile(outdir,[fname '.png']));
        
        fprintf('  Saved: %s.pdf and .png\n', fname);
    end
    
    % Close figures to free memory
    close(figs);
end
