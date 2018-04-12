function [annolist_flat,single_person_flat] = flatten_annolist(annolist,single_person)

annolist_flat = struct('image',[],'annorect',[]);

n = 0;
single_person_flat = [];
for imgidx = 1:length(annolist)
    rect_gt = annolist(imgidx).annorect;
    for ridx = 1:length(rect_gt)
        if (isfield(rect_gt(ridx),'objpos') && ~isempty(rect_gt(ridx).objpos))
            n = n + 1;
            annolist_flat(n).image.name = annolist(imgidx).image.name;
            annolist_flat(n).annorect = rect_gt(ridx);
            single_person_flat(n) = ismember(ridx,single_person{imgidx});
        end
    end
end

end