clear all;
data = load ('data/phy_train.dat');
Y = data(:,2);
X = data(:,[3:21,25:30,32:45,49:56,58:size(data,2)]);
[Xtr,Xte,Ytr,Yte] = splitData(X,Y,0.8);

nt = 500;
%ens = fitensemble(Xtr,Ytr,'AdaBoostM1',nt,'Tree');
%yhat = predict(ens,Xte);
%e = mean( double(Yte ~= yhat) );
%e = 0.2875

ens = fitensemble(X,Y,'AdaBoostM1',nt,'Tree');

data = load('data/phy_test.dat');
Xblind =  data(:,[3:21,25:30,32:45,49:56,58:size(data,2)]);
Yblind = predict(ens,Xblind);
submission = [data(:,1),  Yblind];
dlmwrite('results/adaboost.txt',submission,'delimiter',' ', 'precision',10); 
