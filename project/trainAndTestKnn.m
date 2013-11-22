function [knn K testErr trainErr] = trainAndTestKnn(X, Y)
    [Xtr,Xte,Ytr,Yte] = splitData(X,Y,0.8);
    knn = knnClassify(Xtr, Ytr);
    i = 1;
    K = [1 5:5:40];
    for k = K,
        k
        knn = setK(knn, k);
        yhat = predict(knn, Xte);
        testErr(i) = mean(yhat~=Yte);
        testErr(i)
        yhat = predict(knn, Xtr);
        trainErr(i) = mean(yhat~=Ytr);
        trainErr(i)
        i = i + 1;
    end;
end
