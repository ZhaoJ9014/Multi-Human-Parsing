function pck = computePCK(dist,range)

pck = zeros(numel(range),size(dist,2)+2);

for jidx = 1:size(dist,2)
    % compute PCK for each threshold
    for k = 1:numel(range)
        d = squeeze(dist(1,jidx,:));
        % dist is NaN if gt is missing; ignore dist in this case
        pck(k,jidx) = 100*mean(d(~isnan(d))<=range(k));
    end
end

% compute average PCK upper body
for k = 1:numel(range)
    d = reshape(squeeze(dist(1,7:12,:)),6*size(dist,3),1);
    pck(k,end-1) = 100*mean(d(~isnan(d))<=range(k));
end

% compute average PCK full body
for k = 1:numel(range)
    d = reshape(squeeze(dist(1,:,:)),size(dist,2)*size(dist,3),1);
    pck(k,end) = 100*mean(d(~isnan(d))<=range(k));
end

end