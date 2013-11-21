function obj=train(obj, X,Y, Nbag, Nuse, model, varargin)
%Nbag
%   Nbag: number of classifiers
%   Nuse: number of sample per bag
    N = size(X,1); 
    obj.Classifiers = cell(1,Nbag); % Allocate space 
    for i=1:Nbag 
     ind = ceil( N*rand(Nuse, 1) ); % Bootstrap sample data 
     Xi = X(ind, :); Yi = Y(ind, :); % Select those indices 
     obj.Classifiers{i} = train(model,Xi, Yi, varargin{:}); % Train 
    end; 

end

