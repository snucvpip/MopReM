%% initialize model
PatchSize = 8;
nmodels = 10;

MiniBatchSize = 1500;
filename = sprintf('GMM_%dx%d_%d_%d',PatchSize,PatchSize,nmodels,MiniBatchSize);

% load images into cell
Images = {};
Images{1} = im2double(rgb2gray(imread('1.jpg')));
Images{2} = im2double(rgb2gray(imread('2.jpg')));
Images{3} = im2double(rgb2gray(imread('3.jpg')));
    

%% learn model from training data
NewGMM = OnlineGMMEM(nmodels,@(N) removeDC(RandPatchesFromImagesCell(N,PatchSize,Images)),1000,MiniBatchSize,filename);

% sort output 
[NewGMM.mixweights,inds] = sort(NewGMM.mixweights,'descend');
NewGMM.covs = NewGMM.covs(:,:,inds);