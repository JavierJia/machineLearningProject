clear all;
data = load ('data/phy_train.dat');
Y = data(:,2);
X = data(:,[3:21,25:30,32:45,49:56,58:size(data,2)]);
[Xtr,Xte,Ytr,Yte] = splitData(X,Y,0.8);

Nmin = 100;
DepthMax = Inf;
VarMin = 0.0001;
nFeat = Inf;

[n,mm] = size(Xtr);

for depth = round(2000/100.0 * logspace(0,2));
    tc = treeClassify(Xtr,Ytr,Nmin,depth,VarMin, nFeat);
    errors_depth(depth,1) = err(tc,Xtr,Ytr);
    errors_depth(depth,2) = err(tc,Xte,Yte);
end
figure;
plot (find(errors_depth(:,2)),errors_depth(find(errors_depth(:,2)),2));
[m,id] = min(errors_depth(find(errors_depth(:,2)),2));
t = find(errors_depth(:,2));
m
depth = t(id)


for nmin = round(n/100.0 * logspace(0,2));
    tc = treeClassify(Xtr,Ytr,nmin,depth,VarMin,nFeat);
    errors_nmin(nmin,1) = err(tc,Xtr,Ytr);
    errors_nmin(nmin,2) = err(tc,Xte,Yte);
end
figure;
plot (find(errors_nmin(:,2)),errors_nmin(find(errors_nmin(:,2)),2));
[m,id] = min(errors_nmin(find(errors_nmin(:,2)),2));
t = find(errors_nmin(:,2));
m
nmin = t(id)

for nfeat = round(mm/100.0 * logspace(0,2));
    tc = treeClassify(Xtr,Ytr,nmin,depth,VarMin,nfeat);
    errors_nfeat(nfeat,1) = err(tc,Xtr,Ytr);
    errors_nfeat(nfeat,2) = err(tc,Xte,Yte);
end

figure;
plot (find(errors_nfeat(:,2)),errors_nfeat(find(errors_nfeat(:,2)),2));
[m,id] = min(errors_nfeat(find(errors_nfeat(:,2)),2));
t = find(errors_nfeat(:,2));
m
nfeat = t(id)
