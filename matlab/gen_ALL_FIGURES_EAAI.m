function gen_ALL_FIGURES_EAAI()
clc;
fprintf('==============================================\n');
fprintf('  EAAI PAPER — ALL FIGURES (FIXED V18)\n');
fprintf('==============================================\n\n');
t_total=tic; failed={};

figs = {@gen_fig1_singular_bvp, @gen_fig2_pantograph, ...
        @gen_fig3_riccati_trace, @gen_fig4_loss_evolution, ...
        @gen_fig5_aerospace_trajectory};
names = {'Fig1 Singular BVP','Fig2 Pantograph',...
         'Fig3 Riccati Trace','Fig4 Loss Evolution',...
         'Fig5 Aerospace 6D'};

for k=1:numel(figs)
    fprintf('--- %s ---\n',names{k});
    t=tic;
    try
        figs{k}();
        fprintf('Done in %.1f min\n\n',toc(t)/60);
    catch ME
        fprintf('*** FAILED: %s\n\n',ME.message);
        failed{end+1}=names{k};
    end
end

fprintf('==============================================\n');
fprintf('Total time: %.1f min\n',toc(t_total)/60);
if isempty(failed)
    fprintf('ALL 5 FUNCTIONS SUCCEEDED\n');
else
    fprintf('%d FAILED: ',numel(failed));
    fprintf('%s  ',failed{:}); fprintf('\n');
end
fprintf('==============================================\n');
end