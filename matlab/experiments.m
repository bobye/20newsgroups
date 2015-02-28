function [] = experiments(vocab_name, version)

  db=load_data(['../vocab/' vocab_name], version);

  addpath('liblinear-1.96/matlab');

  train_d = diag(1./(sum(db.train_vec>0,2)+eps))* (db.train_vec>0) * db.vocab_mat(1:size(db.train_vec,2),:);
  test_d  = diag(1./(sum(db.test_vec>0,2)+eps)) * (db.test_vec>0)  * db.vocab_mat(1:size(db.test_vec,2),:);

  disp('classifying ... ');
  disp(['average approach with dimension ', num2str(size(db.vocab_mat,2))]);fflush(stdout);
  classify(sparse(train_d), db.train_lab, sparse(test_d), db.test_lab);


  disp(['raw BoW approach with vocabulary size ', num2str(sum(sum(db.vocab_mat, 2) ~= 0))]);fflush(stdout);
  classify(db.train_vec, db.train_lab, db.test_vec, db.test_lab);

  disp('clustering ...');fflush(stdout);

  data=[train_d; test_d];
  labels=[db.train_lab; db.test_lab];

#  macro_labels = [1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 7, 7, 7, 7];
#  labels=macro_labels(labels);
#  data=data(1:2000,:);
#  labels=labels(1:2000);

#  clustering_evaluation(data, labels, 2);
#  clustering_evaluation(data, labels, 3);
#  clustering_evaluation(data, labels, 4);
#  clustering_evaluation(data, labels, 5);
#  clustering_evaluation(data, labels, 6);
  clustering_evaluation(data, labels, 7);
  clustering_evaluation(data, labels, 10);
  clustering_evaluation(data, labels, 15);
  clustering_evaluation(data, labels, 20);
  clustering_evaluation(data, labels, 30);
  clustering_evaluation(data, labels, 40);
  
end

function [db] = load_data(vocab_name, version)
  db.vocab_mat=load(vocab_name);
  disp('load vocabulary mat');fflush(stdout);

  train_vec=load(['../data/train_tfidf' version '.data']);
  db.train_vec=sparse(train_vec(:,1),train_vec(:,2),train_vec(:,3));
  db.train_lab=load('../data/train.label');
  disp('load tfidf training set');fflush(stdout);
  
  test_vec=load(['../data/test_tfidf' version '.data']);
  db.test_vec=sparse(test_vec(:,1),test_vec(:,2),test_vec(:,3));
  db.test_lab=load('../data/test.label');
  disp('load tfidf test set');fflush(stdout);

  db.train_size = size(train_vec,1);
  db.test_size = size(test_vec,1);
end

function clustering_evaluation(data, labels, k)
  groups = litekmeans(data', k);
%  for i=1:k sum(groups==i) end
  amik= ami(labels, groups);
  adjrandk = adjrand(labels, groups);
  disp(['AMI & ARI with ' num2str(k) ' clusters: ', num2str(amik), ' ', num2str(adjrandk)]); fflush(stdout);
end
