function annolist_orig = unflatten_annolist(annolist_flat,annolist_orig)

n = 0;
for imgidx = 1:length(annolist_orig)
    rect_orig = annolist_orig(imgidx).annorect;
    for ridx = 1:length(rect_orig)
        if (isfield(rect_orig(ridx),'objpos') && ~isempty(rect_orig(ridx).objpos))
            n = n + 1;
            rect_orig(ridx) = annolist_flat(n).annorect;
        end
    end
    annolist_orig(imgidx).annorect = rect_orig;
end

end