function Yhat = predictUnlabeledData_KNN(knn, k, Xunlabed)
    knn = setK(knn, k);
    Yhat = predict(knn, Xunlabed);
end