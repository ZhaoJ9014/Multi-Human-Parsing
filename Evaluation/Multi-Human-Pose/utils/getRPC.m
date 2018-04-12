function [precision, recall, sorted_scores, sortidx, sorted_labels] = getRPC(class_margin, true_labels, totalpos)

N = length(true_labels);
ndet = N;

npos = 0;

[sorted_scores, sortidx] = sort(class_margin, 'descend');
sorted_labels = true_labels(sortidx);

recall = zeros(ndet, 1);
precision = zeros(ndet, 1);

for ridx = 1:ndet
    if sorted_labels(ridx) == 1
        npos = npos + 1;
    end
    
    precision(ridx) = npos / ridx;
    recall(ridx) = npos / totalpos;
end

