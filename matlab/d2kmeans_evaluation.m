filename='../20newsgroups_clean/20newsgroups';
groups=load([filename '.d2s_62308.label_o'])+1;
labels=load([filename '.label']);

ugrp_size=length(unique(groups));

fprintf(stdout, "clusters:%d\nAMI: %f\nARI: %f\n", ugrp_size, ami(labels, groups), adjrand(labels, groups));


