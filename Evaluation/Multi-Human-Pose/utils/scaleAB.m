function X = scaleAB(X,A,B,d)
% X = scaleAB(X,A,B)
% X = scaleAB(X,A,B,d)

if isempty(X), return; end
if nargin == 3
    m = min(X(:));
    M = max(X(:));
    diffval = m-M;
    diffval(diffval==0) = 1;
    X = (X - M)/diffval*(A-B)+B;
else
    m = min(X,[],d);
    M = max(X,[],d);
    diffvals = m - M;
    diffvals(diffvals==0) = 1;
    X = bsxfun(@times,bsxfun(@minus,X,M),1./(diffvals)*(A-B))+ B;
end
0;