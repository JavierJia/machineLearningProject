function e = err(obj,Xte,Yte,NuseBag)
% e = err(obj,Xtest,Ytest)
% calculate the misclassification error for the classifier
    e = mean( double(Yte ~= predict(obj,Xte,NuseBag)) );  
end

