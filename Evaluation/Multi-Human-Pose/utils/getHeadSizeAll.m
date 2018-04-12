function headSizeAll = getHeadSizeAll(annolist)
headSizeAll = nan(length(annolist),1);
for imgidx = 1:length(annolist)
    rect = annolist(imgidx).annorect;
    headSizeAll(imgidx) =  util_get_head_size(rect);
end
end