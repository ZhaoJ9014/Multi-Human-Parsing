function auc = area_under_curve(xpts,ypts)

if nargin == 1
    ypts = xpts;
    xpts = (1:size(ypts,2))/size(ypts,2);
end

a = min(xpts);
b = max(xpts);
%remove duplicate points
[ignore,I,J] = unique(xpts);
xpts = xpts(I);
ypts = ypts(I);
if length(xpts) < 2; auc = NaN; return; end
myfun = @(x)(interp1(xpts,ypts,x));
auc = quadgk(myfun,a,b);

return

%% test

x = (0:0.01:1);
y = x.^2;

figure(1)
plot(x,y)
axis equal

%should integrate to 1/3
auc = area_under_curve(x,y)
