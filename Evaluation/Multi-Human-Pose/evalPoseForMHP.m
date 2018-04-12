%% Script for evaluating mAP for MHP dataset

%% Add path for utils
addpath('./utils');

%% Welcome msg
fprintf('Evaluate Multi-Person Pose Estimation for MHP Dataset()\n');

%% Load ground-truth and predictions
load('./gt/annolist_test_multi_mhp.mat', 'annolist_test_multi');
load('./pred/mhp_pose_results.mat','pred');

%% Hyper-parameters
thresh = 0:0.01:0.5;
tableTex = cell(1,1);
p = getExpParams(0);

%% Calculate mAP
assert(length(annolist_test_multi) == length(pred));
[scoresAll, labelsAll, nGTall] = assignGTmulti(pred,annolist_test_multi,thresh(end));
ap = zeros(size(nGTall,1)+1,1);
for j = 1:size(nGTall,1)
    scores = []; labels = [];
    for imgidx = 1:length(annolist_test_multi)
        scores = [scores; scoresAll{j}{imgidx}];
        labels = [labels; labelsAll{j}{imgidx}];
    end
    [precision,recall] = getRPC(scores,labels,sum(nGTall(j,:)));
    ap(j) = VOCap(recall,precision)*100;
end
ap(end) = mean(ap(1:end-1));

%% Display results
genTableAP(ap,p.name);