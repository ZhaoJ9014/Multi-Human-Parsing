function joints = annolist2matrix(annolist)

joints = nan(2,14,length(annolist));
nJointsAnnolist = 16;

n = 0;
for imgidx = 1:length(annolist)
    pointsAll = nan(nJointsAnnolist,2);
    if (isfield(annolist(imgidx).annorect,'annopoints') && ~isempty(annolist(imgidx).annorect.annopoints) && ...
        isfield(annolist(imgidx).annorect.annopoints,'point') && ~isempty(annolist(imgidx).annorect.annopoints.point))
        points = annolist(imgidx).annorect.annopoints.point;
        for kidx = 0:nJointsAnnolist-1
            p = util_get_annopoint_by_id(points,kidx);
            if (~isempty(p))
                pointsAll(kidx+1,:) = [p.x p.y];
            end
        end
    else
        n = n + 1;
    end
    joints(:,1:6,imgidx) = pointsAll(1:6,:)';
    joints(:,7:12,imgidx) = pointsAll(11:16,:)';
    joints(:,13:14,imgidx) = pointsAll(9:10,:)';
end
fprintf('# missing poses: %d\n',n);
end