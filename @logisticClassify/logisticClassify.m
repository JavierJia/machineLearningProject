function [obj,err,J] = logisticClassify(Xtr,Ytr, varargin)
% perceptClassify( [X,Y,...] ) : create a perceptron classifier and optionally call "train"
  obj.wts=[];         % linear weights on features (1st weight is constant term)
  obj.classes=[];     % list of class values used in input
  obj=class(obj,'logisticClassify');
  if (nargin > 0) 
    [obj,err,J]=train(obj,Xtr,Ytr, varargin{:});
  end;
end

