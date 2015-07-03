function [p]=purity(label, idx)
label=label(:);
idx=idx(:);
assert(length(label) == length(idx));
labelset = unique(label);
idxset = unique(idx);
p=sum(arrayfun(@(i) max(histc(label(idx==i),labelset)), idxset)) / length(label);
