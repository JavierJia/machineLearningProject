clear all;
data = load ('data/filled1000.txt');

Y = data(1:50000,2);
X = data(1:50000,3:end);
[Xtr,Xte,Ytr,Yte] = splitData(X,Y,0.8);

nt = 200;
ens = fitensemble(Xtr,Ytr,'AdaBoostM1',nt,'Tree');
yhat = predict(ens,Xte);
e = mean( double(Yte ~= yhat) );
e 

% ens = fitensemble(X,Y,'AdaBoostM1',nt,'Tree');
% 
% data = load('data/phy_test.dat');
% Xblind =  data(:,[3:21,25:30,32:45,49:56,58:size(data,2)]);
% Yblind = predict(ens,Xblind);
% submission = [data(:,1),  Yblind];
% dlmwrite('results/adaboost.txt',submission,'delimiter',' ', 'precision',10); 
