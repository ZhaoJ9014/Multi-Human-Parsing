function [scoresAll, labelsAll, nGTall] = assignGTmulti(pred,annolist_gt,thresh)

nJoints = 14;
% LSP to MPII format map
jidxMap = [0:5 10:15 8:9];
assert(length(jidxMap) == nJoints);

% part detections scores
scoresAll = cell(nJoints,1);
% positive / negative labels
labelsAll = cell(nJoints,1);
% number of annotated GT joints per image
nGTall = zeros(nJoints,length(annolist_gt));

for i = 1:nJoints
    scoresAll{i} = cell(length(annolist_gt),1);
    labelsAll{i} = cell(length(annolist_gt),1);
end

for imgidx = 1:length(annolist_gt)
    %fprintf('Test image %d / %d\n', imgidx, length(annolist_gt));
    % distance between predicted and GT joints
    dist = inf(length(pred(imgidx).annorect),length(annolist_gt(imgidx).annorect),nJoints);
    % score of the predicted joint
    score = nan(length(pred(imgidx).annorect),nJoints);
    % body joint prediction exist
    hasPred = false(length(pred(imgidx).annorect),nJoints);
    % body joint is annotated
    hasGT = false(length(annolist_gt(imgidx).annorect),nJoints);
    
    % iterate over predicted poses
    for ridxPred = 1:length(pred(imgidx).annorect)
        % predicted pose
        %fprintf('im idx: %d, rect idx: %d\n', imgidx, ridxPred);
        
        rectPred = pred(imgidx).annorect(ridxPred);
        pointsPred = rectPred.annopoints.point;
        % iterate over GT poses
        for ridxGT = 1:length(annolist_gt(imgidx).annorect) % GT
            % GT pose
            rectGT = annolist_gt(imgidx).annorect(ridxGT);
            % compute reference distance as head size
            refDist = util_get_head_size(rectGT);
            pointsGT = [];
            if (~isempty(rectGT.annopoints))
                pointsGT = rectGT.annopoints.point;
            end
            % iterate over all possible body joints
            for i = 1:nJoints
                % predicted joint in LSP format
                ppPred = util_get_annopoint_by_id(pointsPred, jidxMap(i));
                if (~isempty(ppPred))
                    score(ridxPred,i) = ppPred.score;
                    hasPred(ridxPred,i) = true;
                end
                % GT joint in LSP format
                ppGT = util_get_annopoint_by_id(pointsGT, jidxMap(i));
                if (~isempty(ppGT))
                    hasGT(ridxGT,i) = true;
                end
                % compute distance between predicted and GT joint locations
                if (hasPred(ridxPred,i) && hasGT(ridxGT,i))
                    dist(ridxPred,ridxGT,i) = norm([ppGT.x ppGT.y] - [ppPred.x ppPred.y])/refDist;
                end
            end % joints
        end % GT poses
    end % predicted poses
    
    % number of annotated joints
    nGT = repmat(sum(hasGT,2)',size(hasPred,1),1);
    % compute PCKh
    match = dist <= thresh;
    pck = sum(match,3)./nGT;
    % preserve best GT match only
    [val,idx] = max(pck,[],2);
    for ridxPred = 1:length(idx)
        pck(ridxPred,setdiff(1:size(pck,2),idx(ridxPred))) = 0;
    end
    [val,predToGT] = max(pck,[],1);
    predToGT(val == 0) = 0;
    
    % assign predicted poses to GT poses
    for ridxPred = 1:length(pred(imgidx).annorect)
        if (ismember(ridxPred,predToGT)) % pose matches to GT
            % GT pose that matches the predicted pose
            ridxGT = find(predToGT == ridxPred);
            s = score(ridxPred,:);
            m = squeeze(match(ridxPred,ridxGT,:));
            hp = hasPred(ridxPred,:);
            idxs = find(hp);
            for i = 1:length(idxs)
                scoresAll{idxs(i)}{imgidx} = [scoresAll{idxs(i)}{imgidx};s(idxs(i))];
                labelsAll{idxs(i)}{imgidx} = [labelsAll{idxs(i)}{imgidx};m(idxs(i))];
            end
        else % no matching to GT
            s = score(ridxPred,:);
            m = false(size(match,3),1);
            hp = hasPred(ridxPred,:);
            idxs = find(hp);
            for i = 1:length(idxs)
                scoresAll{idxs(i)}{imgidx} = [scoresAll{idxs(i)}{imgidx};s(idxs(i))];
                labelsAll{idxs(i)}{imgidx} = [labelsAll{idxs(i)}{imgidx};m(idxs(i))];
            end
        end
    end % prediction assignment
    
    % save number of GT joints
    for ridxGT = 1:length(annolist_gt(imgidx).annorect)
        hg = hasGT(ridxGT,:);
        nGTall(:,imgidx) = nGTall(:,imgidx) + hg';
    end
end
end