function s = sig(z)
% value of [0,1] sigmoid
    s = 1 ./ (1+exp(-z));
end

