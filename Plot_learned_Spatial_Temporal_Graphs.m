%%
clear;
close all;
clc;
%%
Path_folder = '.\MASS_SS3_Results_folder\';
LineWidth = 1.75;
%% EEG Spatial graphs:
load('.\ChanLocs_and_ChanNames.mat');

fig = figure('units', 'normalized', 'outerposition', [0 0 1 1]);

SpatialGraphs = readNPY([Path_folder,'MASS_SS3_LearnedGraphsSpatial.npy']);

Labels = double(readNPY([Path_folder,'MASS_SS3_Labels_All.npy']));

MeanGraphs = zeros(5, 26, 26);

for stage = 1:5

    Graphs = SpatialGraphs(Labels == stage-1, :, :);
    
    MeanGraphs(stage,:,:) = squeeze(mean(Graphs, 1));
end


MeanGraphs_nonZero = MeanGraphs(MeanGraphs~=0);
MeanGraphs = (MeanGraphs - min(MeanGraphs_nonZero(:)))/(max(MeanGraphs(:)) - min(MeanGraphs_nonZero(:)));

title_vec = {'W', 'N1', 'N2', 'N3', 'R'};

Thr = 0.4;

for i = 1:5

    A = squeeze(MeanGraphs(i,:,:));
    
    A = (A+A')/2;
        
%     A( A < Thr*max(A(:))) = 0;
    
    A( A < Thr) = 0;

    A = A(3:22, 3:22);

    G = graph(A,'omitselfloops');
    
    x = ChanLocs(1:end-1,1)';
    
    y = ChanLocs(1:end-1,2)';
    
    ChanNames = ChanNames(1, 1:20);

    subplot(3,5,i); plot(G,'XData',x,'YData',y, 'NodeLabel',ChanNames, 'LineWidth', LineWidth, 'NodeColor', 'g', 'EdgeColor','g'); 
    
    camroll(90)
        
    title([title_vec{i}, ', Thr: ', num2str(Thr)])
    
    xticks([]); xticklabels({''});
    
    yticks([]); yticklabels({''});
    
    color = get(fig,'Color');

    set(gca,'XColor',color,'YColor',color,'TickDir','out')
    
    if i == 3
        
        ylabel('(a) Learned spatial EEG graphs', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');
        
    end


end
%% TemporalGraphs:
TemporalGraphs = readNPY([Path_folder,'MASS_SS3_LearnedGraphsTempral.npy']);

% for fold = 1:15
%     
%     Labels = cat(1, Labels, readNPY([Path_folder,'TestLabels_fold',num2str(fold),'.npy']));
%         
%     TemporalGraphs = cat(1, TemporalGraphs, readNPY([Path_folder,'LearnedGraphsTempral_fold',num2str(fold),'.npy']));   
% 
% end

title_vec = {'W', 'N1', 'N2', 'N3', 'R'};

Thr = 0.3;

MeanGraphs = zeros(5,9,9);

for stage = 1:5

    Graphs = TemporalGraphs(Labels == stage-1, :, :);
    
    MeanGraphs(stage,:,:) = squeeze(mean(Graphs, 1));
end


% MeanGraphs = (MeanGraphs - min(MeanGraphs(:)))/(max(MeanGraphs(:)) - min(MeanGraphs(:)));

for stage = 1:5
   
    A = squeeze(MeanGraphs(stage,:,:));
    
    A = (A+A')/2;
        
    A = A - diag(diag(A));
    
    B = nonzeros(triu(A));
    
    A = (A - min(B(:)))/(max(B(:)) - min(B(:)));

%     A( A < Thr*max(A(:))) = 0;
    
    A( A < Thr) = 0;
    
    G = graph(A,'omitselfloops');
        
    LWidths = 3*G.Edges.Weight/max(G.Edges.Weight);

    ChanNames_temporal = {'t-4', 't-3', 't-2', 't-1', 'target', 't+1', 't+2', 't+3', 't+4'};

    subplot(3,5,stage + 10); plot(G, 'NodeLabel',ChanNames_temporal, 'Layout', 'circle', 'LineWidth', LineWidth, 'NodeColor', 'b',...
        'EdgeColor','b'); 
    
    title([title_vec{stage}, ', Thr: ', num2str(Thr)])
    
    xticks([]); xticklabels({''});
    
    yticks([]); yticklabels({''});
    
    color = get(fig,'Color');

    set(gca,'XColor',color,'YColor',color,'TickDir','out')
    
    if stage == 3
        
        xlabel('(c) Learned temporal graphs', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');
        
    end


end

%% Neighbor Histograms:
% Neighbors = {[],[],[],[],[]};
% 
% for i  = 5 : length(Labels) - 4
%     
%     Neighbors{Labels(i) + 1} = [Neighbors{Labels(i) + 1}; [Labels(i - 4 : i - 1)' , Labels(i + 1 : i + 4)']];
%         
% end
% 
% fig = figure('units', 'normalized', 'outerposition', [0 0 1 1]);
% 
% title2 = {'t-4', 't-3', 't-2', 't-1', 't+1', 't+2', 't+3', 't+4'};
% 
% stage_name = {'W', 'N1', 'N2', 'N3', 'R'};
% 
% ylimMat = [0, 4000; 0, 3000; 0, 30000; 0, 6000; 0, 10000];
% 
% for stage = 0 : 4
%     
%     matr = Neighbors{stage + 1};
%     
%     for i = 1 : 8
% 
%         subplot(5, 8, stage*8 + i); histogram(matr(:, i)); title([stage_name{stage+1},', ', title2{i}], 'FontSize', 12); ylim(ylimMat(stage+1,:));
% 
%         xticks([0, 1, 2, 3, 4]); xticklabels({'0', '1', '2', '3', '4'});
%     
% %     yticks([]); yticklabels({''});
% 
%     end
% 
% end


%% All channels Spatial graphs:
clc;

load('.\ChanLocs_and_ChanNames.mat');

% fig = figure('units', 'normalized', 'outerposition', [0 0 1 1]);

SpatialGraphs = readNPY([Path_folder,'MASS_SS3_LearnedGraphsSpatial.npy']);

% for fold = 1:15
%         
%     Labels = cat(1, Labels, readNPY([Path_folder,'TestLabels_fold',num2str(fold),'.npy']));
%     SpatialGraphs = cat(1, SpatialGraphs, readNPY([Path_folder,'LearnedGraphsSpatial_fold',num2str(fold),'.npy']));   
%     
% end

MeanGraphs = zeros(5, 26, 26);

for stage = 1:5

    Graphs = SpatialGraphs(Labels == stage-1, :, :);
    
    MeanGraphs(stage,:,:) = squeeze(mean(Graphs, 1));
end


MeanGraphs_nonZero = MeanGraphs(MeanGraphs~=0);
MeanGraphs = (MeanGraphs - min(MeanGraphs_nonZero(:)))/(max(MeanGraphs(:)) - min(MeanGraphs_nonZero(:)));

title_vec = {'W', 'N1', 'N2', 'N3', 'R'};

Thr = 0.4;

for i = 1:5

    A = squeeze(MeanGraphs(i,:,:));
    
    A = (A+A')/2;
        
%     A( A < Thr*max(A(:))) = 0;
    
    A( A < Thr) = 0;
        
    A(3:22, 3:22) = 0;
    
    A = (A+A')/2;

%     A = A(1:25, 1:25);

    G = graph(A,'omitselfloops');
    
    x = [ChanLocs(2,1)+0.05, ChanLocs(1,1)+0.05, ChanLocs(1:end-1,1)', ChanLocs(15,1)-0.05,...
        ChanLocs(16,1)-0.05, ChanLocs(20,1)-0.05, ChanLocs(20,1)-0.1];
    
    y = [ChanLocs(2,2)-0.05, ChanLocs(1,2)+0.05, ChanLocs(1:end-1,2)', ChanLocs(15,2)+0.05,...
        ChanLocs(16,2)-0.05, ChanLocs(20,2), ChanLocs(20,2)];
    
    ChanNames2 = {{'{EOG-R}','{EOG-L}'}, ChanNames(1, 1:20), {'{EMG-1}','{EMG-2}', '{EMG-3}', 'ECG'}};
    
    ChanNames2 = cat(2, ChanNames2{:});

    LWidths = 2*G.Edges.Weight/max(G.Edges.Weight);

    subplot(3,5,i+5); plot(G,'XData',x,'YData',y, 'NodeLabel',ChanNames2, 'LineWidth', LineWidth, 'NodeColor', 'r', ...
        'EdgeColor','r'); 
    
    camroll(90)
        
    title([title_vec{i}, ', Thr: ', num2str(Thr)])
    
    xticks([]); xticklabels({''});
    
    yticks([]); yticklabels({''});
    
    color = get(fig,'Color');

    set(gca,'XColor',color,'YColor',color,'TickDir','out')
    
    if i == 3
        
        ylabel('(b) Learned spatial non-EEG graphs', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');
        
    end


end


