function dist = getDistPCKh(pred,gt,refDist)

assert(size(pred,1) == size(gt,1) && size(pred,2) == size(gt,2) && size(pred,3) == size(gt,3));
assert(size(refDist,1) == size(gt,3));

dist = nan(1,size(pred,2),size(pred,3));

for imgidx = 1:size(pred,3)
    
    % distance to gt joints
    dist(1,:,imgidx) = sqrt(sum((pred(:,:,imgidx) - gt(:,:,imgidx)).^2,1))./refDist(imgidx);

end