function [] = classify(train_vec, train_lab, test_vec, test_lab) 

addpath('liblinear-1.96/matlab');

model=train(train_lab,train_vec, '-q');
[predict_label, accuracy, prob_estimates] = predict(test_lab, test_vec, model);

recal=0;
precision=0;
for i=1:20
    recal = recal + sum(predict_label(test_lab==i)==i) / sum(test_lab==i);
    precision = precision + sum(test_lab(predict_label==i)==i) / sum(predict_label==i);
end
recal=recal/20
precision=precision/20
