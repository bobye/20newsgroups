function [] = matrixtod2s(name, version, dim, isSemiSup)

  tag=version;
  if (isSemiSup)
    tag = [tag 's'];
  end
  data_vec=load(['../data/' name '_tfidf' tag '.data']);
  data_vec=sparse(data_vec(:,1),data_vec(:,2),data_vec(:,3));

  filename=['../d2s/' name '_tfidf' tag '.d2s'];
  f=fopen(filename,'w');
  for i=1:size(data_vec,1)
    wordids=find(data_vec(i,:));
    #  assert(length(wordids) > 0);
    if (length(wordids) == 0) 
      wordids=[0]; # if empty
      weight=1.0;
    else
      weight = data_vec(i,wordids); weight = weight/sum(weight);      
    end
    fprintf(f, '%d\n%d\n', dim, length(wordids));
    fprintf(f, '%f ', weight);
    fprintf(f, '\n');
    fprintf(f, '%d ', wordids);
    fprintf(f, '\n');  
    fprintf(f, '\n');  
  end
  fclose(f);
  disp(['save to ' filename]);fflush(stdout);

end
