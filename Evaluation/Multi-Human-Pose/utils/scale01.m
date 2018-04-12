function X = scale01(X,d)
X = double(X);
if nargin == 1
    X = scaleAB(X,0,1);
else
    X = scaleAB(X,0,1,d);
end

return

% if isempty(X), return; end
% if nargin == 1
%     minval = min(X(:));
%     maxval = max(X(:));
%     diffval = maxval-minval;
%     diffval(diffval==0) = 1;
%     X = (X - minval)/diffval;
% else
%     minvals = min(X,[],d);
%     maxvals = max(X,[],d);
%     diffvals = maxvals - minvals;
%     diffvals(diffvals==0) = 1;
%     X = bsxfun(@times,bsxfun(@minus,X,minvals),1./(diffvals));
% end
0;