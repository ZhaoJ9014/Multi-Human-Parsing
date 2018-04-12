%% Draw curve for multi stage

figure(100); clf; hold on;
set(0,'DefaultAxesFontSize',16);
set(0,'DefaultTextFontSize',16);



range = 0:0.01:0.5;

stages = [8, 4, 2, 1];

legendName = cell(length(stages),1);

colors = [228, 26, 28;
          55, 126, 184;
          245, 170, 27;
          128, 128, 225];

colors = colors  / 255;
      
for ti = 1:length(stages)
    si = stages(ti);
    mat_name = sprintf('./exp/ap_list_stage%d.mat', si);
    ap_list_struct = load(mat_name);
    ap_list = ap_list_struct.ap_list;
    plot(range, ap_list / 100, 'color', colors(ti, :), 'LineStyle','-','LineWidth',3);
    legendName{ti} = sprintf('%d-stage', si);
end

legend(legendName, 'Location', 'SouthEast', 'fontsize', 24);
set(gca,'YLim',[0 1], 'fontsize', 18);
set(gca, 'fontsize', 18);
xlabel('Normalized Distance', 'fontsize', 28);
ylabel('Mean Average Precision', 'fontsize', 28);
grid on;

fname = './exp/multi_stage_ablation_exp';
bSave = 1;
if (bSave)
    print(gcf, '-dpng', [fname '.png']);
    printpdf([fname '.pdf']);
    savefig([fname '.fig']);
end

%% Draw curve for validation acc
% val_acc_on_mpi_hg = load('./val_acc_on_mpi_hg.mat');
% val_acc_on_mpi_hg = val_acc_on_mpi_hg.val_acc;
% val_acc_on_mpi_hg_ltl = load('./val_acc_on_mpi_hg_ltl.mat');
% val_acc_on_mpi_hg_ltl = val_acc_on_mpi_hg_ltl.val_acc;
% 
% epochs = [1 10:10:100 102:2:150 151:250];
% 
% figure(100); clf; hold on;
% set(0,'DefaultAxesFontSize',16);
% set(0,'DefaultTextFontSize',16);
% 
% legendName = cell(2,1);
% 
% plot(epochs, val_acc_on_mpi_hg, 'color', [0, 161, 233] / 255, 'LineStyle','-','LineWidth',3);
% legendName{1} = 'hg';
% plot(epochs, val_acc_on_mpi_hg_ltl, 'color', [233, 161, 0] / 255, 'LineStyle','-','LineWidth',3);
% legendName{2} = 'hg-w-ltl';
% 
% legend(legendName, 'Location', 'SouthEast');
% % set(gca,'YLim',[0 0.14]);
% xlabel('Epoch');
% ylabel('Validation Accuracy');
% grid on;
% 
% fname = './figs/mpi_val_acc_cmp';
% bSave = 1;
% if (bSave)
%     print(gcf, '-dpng', [fname '.png']);
%     printpdf([fname '.pdf']);
%     savefig([fname '.fig']);
% end