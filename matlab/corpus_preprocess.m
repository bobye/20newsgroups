function [] = corpus_preprocess(name, vocab_name, version)

  # format sparse token counts
  data = load(['../data/' name '.data']);
  data_mat=sparse(data(:,1), data(:,2), data(:,3));
  disp(['load ' name '.data']);fflush(stdout);

  # load vocabulary
  vocab = textread('../vocab/vocabulary.txt', '%s');
  vocab_size=length(vocab);
  disp('load vocabulary');fflush(stdout);

  # load stop list and remove them from corpus
  stoplist= textread('../data/stoplist2.txt', '%s');
  disp('load stop words');fflush(stdout);

  for i=1:length(stoplist)
    idx = find(strcmp(stoplist{i},vocab));
    if ~isempty(idx)
      data_mat(:,idx) = 0;
    %    disp(stoplist{i});
    end
  end
  disp('stop words removed');

  # save stopword removed version
  [I, J, S] = find(data_mat');
  data2 = [J, I, S];
  #f=fopen([name, '_preproc0.data'], 'w');
  #fprintf(f, '%d %d %d \n', data2')
  #fclose(f);
  #disp('save version0 (stop words removed)');

  # load distributed representation from googlenews
  vocab_vec=load(['../vocab/' vocab_name]);

  ind=(sum(vocab_vec,2) == 0);
  data_mat(:,ind)=0;
  disp('remove words that donot appear in dictionary');fflush(stdout);

  # save version with noappear version
  [I, J, S] = find(data_mat');
  data2 = [J, I, S];
  f=fopen(['../data/' name, '_preproc' version '.data'], 'w');
  fprintf(f, '%d %d %d \n', data2')
  fclose(f);
  disp(['save version' version ' .data']);fflush(stdout);

  f=fopen(['../mm/' name, '_preproc' version '.mm'], 'w');
  fprintf(f, '%%%%MatrixMarket matrix coordinate real general\n');
  fprintf(f, '%d %d %d \n', [size(data_mat), size(data2, 1)]);
  fprintf(f, '%d %d %d \n', data2')
  fclose(f);
  disp(['save version' version ' .mm']);fflush(stdout);

  #word_count=full(sum(data_mat));
  #[sort_wc, sort_idx] = sort(word_count, 'descend');



