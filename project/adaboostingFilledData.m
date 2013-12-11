clear all;
data = load ('data/filled1000.txt');

Y = data(1:50000,2);
X = data(1:50000,3:end);
X = normolize(X);
[Xtr,Xte,Ytr,Yte] = splitData(X,Y,0.8);

nt = 500;
ens = fitensemble(Xtr,Ytr,'AdaBoostM1',nt,'Tree');
yhat = predict(ens,Xte);
e = mean( double(Yte ~= yhat) );
e 

ens = fitensemble(X,Y,'AdaBoostM1',nt,'Tree');

Xblind =  data(50001:end,3:end);
Xblind = normolize(Xblind);
Yblind = predict(ens,Xblind);
submission = [(50001:150000)',  Yblind];
dlmwrite('results/adaboostfilled.txt',submission,'delimiter',' ', 'precision',10); 
