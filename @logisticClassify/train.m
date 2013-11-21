function [obj,err,J] = train(obj, Xtrain, Ytrain, stepsize, maxSteps)
% obj = train(obj, Xtrain, Ytrain, stepsize, maxSteps)
%     Xtrain = [n x d] training data features (constant feature not included)
%     Ytrain = [n x 1] training data classes (binary, e.g., +1 or -1)
%     stepsize = step size for gradient descent ("learning rate")
%     maxSteps = maximum number of steps before stopping
%
if (nargin < 5) maxSteps = 5000;  end;  % max number of iterations
if (nargin < 4) stepsize = .01;   end;  % gradient descent step size

plotFlag = 1;                    % with plotting

[n,d] = size(Xtrain);            % d = dimension of data; n = number of training data
Xtrain1= [ones(n,1), Xtrain];    % make a version of training data with the constant feature

% Get class id values and replace with values 1..C  
[Ytrain, obj.classes] = toIndex(Ytrain);
if (length(obj.classes)~=2) error('Y values must be binary!'); end;  % check correct binary labeling
%Ytrain = 2*Ytrain-3 ;             % convert Y to (-1,1) for tradition
Ytrain = Ytrain-1 ;             % convert Y to (0,1) for tradition

obj.wts = randn(1,d+1);          % initialize weights randomly

% Outer loop of stochastic gradient descent:
iter=1;                          % iteration #
done=0;                          % end of loop flag
err=zeros(1,maxSteps);           % misclassification rate values
J=zeros(1,maxSteps);
while (~done) 
  % Step size evolution
  stepi = stepsize;              % perceptron method: usually fixed step size

  % Stochastic gradient update (one pass)
  J(iter)=0;
  for i=1:n,  % for each data example,
    resp = Xtrain1(i,:)*obj.wts';                       % compute linear response
    %y01 = 0.5*Ytrain(i)+0.5;
    y01 = Ytrain(i);
    if (sig(resp) >0 && sig(resp) <1);
        J(iter) = J(iter)+(-y01*log(sig(resp)))+(-(1-y01)*log(1-sig(resp)));
    end
    deltaJ = -y01*(1-sig(resp))+(1-y01)*sig(resp);
    deltaJX = deltaJ * Xtrain1(i,:);
    obj.wts = obj.wts - stepi * deltaJX;
  end;

  % Compute current error value
  err(iter)  = mean( (Ytrain~=0.5 + 0.5*sign(Xtrain1*obj.wts')) ); % misclassification rate
  J(iter) = mean(J(iter));
    
  % Make plots, if desired
  if (plotFlag),
  fig(1); plot(1:iter, err(1:iter),'g-');
  
  fig(2); 
    switch d,                              % Plots to help with visualization
      case 1, plot1DLinear(obj,Xtrain,Ytrain);      %  for 1D data we can display the data and the function
      case 2, plot2DLinear(obj,Xtrain,Ytrain);      %  for 2D data, just the data and decision boundary
      otherwise, % no plot for higher dimensions... %  higher dimensions visualization is hard
    end;
    
  fig(3); plot(1:iter, J(1:iter),'b-');
  drawnow;
  end;
  

  done = (iter >= maxSteps || err(iter)==0);        % stop when no errors or out of time
  iter = iter + 1;
end;




