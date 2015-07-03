meta_data{1}.vocab='vocab_vec_googlenews.txt';
meta_data{1}.wordrep='GoogleNews-vectors-negative300.bin';
meta_data{1}.dim = 300;

meta_data{2}.vocab='vocab_vec_wikipedia.txt';
meta_data{2}.wordrep='glove_6B_300d.bin';
meta_data{2}.dim = 300;

meta_data{3}.vocab='vocab_vec100_wikipedia.txt';
meta_data{3}.wordrep='glove_6B_100d.bin';
meta_data{3}.dim = 100;

meta_data{4}.vocab='vocab_400_10_10_wikipedia.txt';
meta_data{4}.wordrep='word2vec_400_10_10.bin';
meta_data{4}.dim = 400;

idx = 3;

% dump vocabulary from pre-trained words representation 
system(['python ../python/dump_vocab.py ' meta_data{idx}.wordrep ' ' meta_data{idx}.vocab]);

% preprocess corpus by removing stopwords and non-appear words
corpus_preprocess('train', meta_data{idx}.vocab, num2str(idx));
corpus_preprocess('test', meta_data{idx}.vocab, num2str(idx));

% compute tfidf and output
system(['python ../python/corpus_preprocess.py ' num2str(idx)]);

% save tfidf data in d2s format
matrixtod2s('train', num2str(idx), meta_data{idx}.dim, false);
matrixtod2s('test', num2str(idx), meta_data{idx}.dim, false)
matrixtod2s('train', num2str(idx), meta_data{idx}.dim, true);
matrixtod2s('test', num2str(idx), meta_data{idx}.dim, true);

% classification
experiments(meta_data{idx}.vocab, num2str(idx));
