clear all;
data = load ('data/filled1000.txt');
train = data(1:50000,:);

Y = train(:,2);
X = train(:,3:size(data,2));
[Xtr,Xte,Ytr,Yte] = splitData(X,Y,0.8);

nc = 200;
b = TreeBagger(nc,Xtr,Ytr,'oobpred','on');
plot(oobError(b))
xlabel('number of grown trees')
ylabel('out-of-bag classification error')

% yhat = str2num(cell2mat(predict(b,Xtr)));
% e = mean( double(Ytr ~= yhat) );
% e


yhat = str2num(cell2mat(predict(b,Xte)));
e = mean( double(Yte ~= yhat) );
e

test = data(50001:end,:);
Xblind = test(:,3:end);
Yblind = str2num(cell2mat(predict(b,Xblind)));
submission = [50001:150000, Yblind];
dlmwrite('results/treebagFilledMissdata.txt',submission,'delimiter',' ', 'precision',10); 
