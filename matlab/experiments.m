function [] = experiments(vocab_name, version)

  db=load_data(['../vocab/' vocab_name], version);

  addpath('liblinear-1.96/matlab');

  train_d = diag(1./(sum(db.train_vec,2)+eps))* (db.train_vec) * db.vocab_mat(1:size(db.train_vec,2),:);
  test_d  = diag(1./(sum(db.test_vec,2)+eps)) * (db.test_vec)  * db.vocab_mat(1:size(db.test_vec,2),:);

  disp('classifying ... ');
  disp(['average approach with dimension ', num2str(size(db.vocab_mat,2))]);fflush(stdout);
  classify(sparse(train_d), db.train_lab, sparse(test_d), db.test_lab);


  disp(['raw BoW approach with vocabulary size ', num2str(sum(sum(db.vocab_mat, 2) ~= 0))]);fflush(stdout);
  classify(db.train_vec, db.train_lab, db.test_vec, db.test_lab);

  disp('clustering ...');fflush(stdout);
  groups = litekmeans([train_d; test_d]', 10);
  nmi10 = nmi([db.train_lab; db.test_lab], groups);
  adjrand10 = adjrand([db.train_lab; db.test_lab], groups);
  disp(['NMI & ARI with 10 clusters: ', num2str(nmi10), ' ', num2str(adjrand10)]); fflush(stdout);


  groups = litekmeans([train_d; test_d]', 20);
  nmi20 = nmi([db.train_lab; db.test_lab], groups);
  adjrand20 = adjrand([db.train_lab; db.test_lab], groups);
  disp(['NMI & ARI with 20 clusters: ', num2str(nmi20), ' ', num2str(adjrand20)]); fflush(stdout);

  groups = litekmeans([train_d; test_d]', 40);
  nmi40 = nmi([db.train_lab; db.test_lab], groups);
  adjrand40 = adjrand([db.train_lab; db.test_lab], groups);
  disp(['NMI & ARI with 40 clusters: ', num2str(nmi40), ' ', num2str(adjrand40)]); fflush(stdout);

  groups = litekmeans([train_d; test_d]', 60);
  nmi60 = nmi([db.train_lab; db.test_lab], groups);
  adjrand60 = adjrand([db.train_lab; db.test_lab], groups);
  disp(['NMI & ARI with 60 clusters: ', num2str(nmi60), ' ', num2str(adjrand60)]); fflush(stdout);

  groups = litekmeans([train_d; test_d]', 100);
  nmi100 = nmi([db.train_lab; db.test_lab], groups);
  adjrand100 = adjrand([db.train_lab; db.test_lab], groups);
  disp(['NMI & ARI with 100 clusters: ', num2str(nmi100), ' ', num2str(adjrand100)]); fflush(stdout);


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
