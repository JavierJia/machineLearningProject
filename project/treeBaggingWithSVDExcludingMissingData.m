clear
data = load ('data/phy_train.dat');
X = data(:,[3:21,25:30,32:45,49:56,58:end]);
Y = data(:,2);
X = X (:,find(abs(sum(X)) > 0));
[u s v] = svds(X, 20);
Xnew = u;
[Xtr,Xte,Ytr,Yte] = splitData(Xnew,Y,0.8);

nc = 100;
b = TreeBagger(nc,Xtr,Ytr,'oobpred','on');
plot(oobError(b))
xlabel('number of grown trees')
ylabel('out-of-bag classification error')

yhat = str2num(cell2mat(predict(b,Xtr)));
e = mean( double(Ytr ~= yhat) );
e


yhat = str2num(cell2mat(predict(b,Xte)));
e = mean( double(Yte ~= yhat) );
e


testdata = load ('data/phy_test.dat');

caseId = testdata(:,1);
Xtest = testdata(:,[3:21,25:30,32:45,49:56,58:end]);
Xtest = Xtest (:,find(abs(sum(Xtest)) > 0));
[ut st vt] = svds(Xtest,20);
%XtestNew = ut *st;
XtestNew = ut;
yhat = str2num(cell2mat(predict(b,XtestNew)));

dlmwrite('results/treebagSVD.txt',[caseId yhat],'delimiter',' ', 'precision',10); 
