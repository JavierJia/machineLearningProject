function [  ] = xinPercept( )
%XINPERCEPT Summary of this function goes here
%   Detailed explanation goes here
    rawData = load('data/phy_test.dat');

    X = rawData(:, [3:21, 25:30, 32:45, 49:56, 58:size(rawData,2)] );
    Y = rawData(:, 2);

    Y(find(Y==0),1) = -1;  % make the zeros to minus one in target value
    [Xtr, Xte, Ytr, Yte] = splitData(X, Y, 0.8);

    stepsize = 0 : 0.2 : 2 ;
    maxStep = 500;
    trainErr = zeros(length(stepsize)-1, 2);
    testErr = zeros(length(stepsize)-1, 2);
    for i = 2 : length(stepsize)
       lc = perceptClassify(Xtr, Ytr, stepsize(i), maxStep);
       trainErr(i-1, 1) = err(lc, Xtr, Ytr);
       testErr(i-1, 1) = err(lc, Xte, Yte);
    end
    [minval, minCol] = min(testErr(:,1));
    optStepSize = stepsize(minCol + 1)
    figure; hold;
    plot(1:10, trainErr, 'b-', 1:10, testErr, 'r-');


    maxStep = 100: 100: 1000;
    stepsize = 1;
    for i = 1 : length(maxStep)
        lc = perceptClassify(Xtr, Ytr, stepsize, maxStep(i));
        trainErr(i, 2) = err(lc, Xtr, Ytr);
        testErr(i, 2) = err(lc, Xte, Yte);
    end
    [minval, minCol] = min(testErr2(:,2));
    optMaxStep = maxStep(minCol)
    figure;hold;
    plot(1:10, trainErr2, 'b-', 1:10, testErr2, 'r-');

end

