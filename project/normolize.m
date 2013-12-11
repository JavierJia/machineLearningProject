function [ normX ] = normolize( X )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
meanX = mean(X);
stdX = std(X);
normX = X;
for i = 1:size(X,2), 
    normX(:,i) = ( X(:,i) - meanX(i) ) / stdX(i); 
end;
end

