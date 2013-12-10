clear all;
dataTrain = load ('data/phy_train.dat');
dataTest = load('data/phy_test.dat');
dataTest(:,2) = NaN;

data = [dataTrain;dataTest];

for i = [22:24,46:48]
    data(find(data(:,i)==999),i) = NaN;
end

for i = [31,57]
    data(find(data(:,i)==9999),i) = NaN;
end

%dlmwrite('data/missingData2NanAll.txt',data,'delimiter','\t');

filled = knnimpute(data, 50,'Distance','euclidean');
dlmwrite('data/filled50.txt',filled,'delimiter','\t');
