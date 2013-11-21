function obj = baggingClassify(Xtr,Ytr, nbag, ndata, model, varargin)

    obj.Classifiers = cell(1,size(Xtr,1));
    obj.Nbag = nbag;
    if (nargin>0)
        obj = train(obj, Xtr,Ytr,nbag, ndata, model, varargin{:});
    end
end