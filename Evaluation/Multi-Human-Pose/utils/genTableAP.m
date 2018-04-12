function [row,header] = genTableAP(ap,name)

assert(length(ap)==15)
header = sprintf(' &Head & Shoulder & Elbow & Wrist & Hip & Knee  & Ankle & Total%s\n','\\');
row = sprintf('%s& %1.1f  & %1.1f  & %1.1f  & %1.1f  & %1.1f  & %1.1f & %1.1f & %1.1f %s\n',name,(ap(13)+ap(14))/2,(ap(9)+ap(10))/2,(ap(8)+ap(11))/2,(ap(7)+ap(12))/2,(ap(3)+ap(4))/2,(ap(2)+ap(5))/2,(ap(1)+ap(6))/2,ap(end),'\\');

fprintf('\n%s %s',blanks(length(name)),header);
fprintf('%s\n',row);
end