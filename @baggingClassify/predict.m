function Yte = predict(obj, Xtest, NuseBag)
% Test data Xtest 
    Ntest = size(Xtest); 
    if (nargin < 3)
        nbag = min(NuseBag, obj.Nbag);
    else
        nbag = obj.Nbag;
    end
    predicts = zeros(Ntest,nbag); % Allocate space 
    for i=1: nbag, % Apply each classifier 
        predicts(:,i)=predict(obj.Classifiers{i}, Xtest); 
    end; 
    Yte = (mean(predicts,2) > 0.5); % Vote on output (0 vs 1)
end

