% Removes DC component from image patches
% Data given as a matrix where each patch is one column vectors
% That is, the patches are vectorized.

function [Y,DC]=removeDC(X)

% Subtract local mean gray-scale value from each patch in X to give output Y
% DC = single(ones(size(X,1),1)*mean(X));
% Y = single(X-DC);
DC = mean(X,1);
Y = bsxfun(@minus,X,DC);
