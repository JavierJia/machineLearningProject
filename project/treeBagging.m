clear all;
data = load ('data/phy_train.dat');
Y = data(:,2);
X = data(:,[3:21,25:30,32:45,49:56,58:size(data,2)]);
[Xtr,Xte,Ytr,Yte] = splitData(X,Y,0.8);

%for nc = round(1000/100.0 * logspace(0,2)),
nc = 100;
b = TreeBagger(nc,Xtr,Ytr,'oobpred','on');
plot(oobError(b))
xlabel('number of grown trees')
ylabel('out-of-bag classification error')

yhat = str2num(cell2mat(predict(b,Xte)));
e = mean( double(Yte ~= yhat) );
e
%errs(nc) =e;   
%end
data = load('data/phy_test.dat');
Xblind =  data(:,[3:21,25:30,32:45,49:56,58:size(data,2)]);
Yblind = str2num(cell2mat(predict(b,Xblind)));
submission = [data(:,1),  Yblind];
dlmwrite('results/treebag.txt',submission,'delimiter',' ', 'precision',10); 

%errs(1,find(errs))
