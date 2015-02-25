filename='../20newsgroups_clean/20newsgroups';
ind=load([filename '.d2s.ind']);
groups_byind=load([filename '.d2s.label']);
labels=load([filename '.label']);

num_of_objects = length(ind);
groups=zeros(num_of_objects,1);
groups(ind+1)=groups_byind+1;
labels = labels(1:num_of_objects);

ugrp_size=length(unique(groups));

fprintf(stdout, "clusters:%d\nNMI: %f\nARI: %f", ugrp_size, nmi(labels, groups), adjrand(labels, groups));

