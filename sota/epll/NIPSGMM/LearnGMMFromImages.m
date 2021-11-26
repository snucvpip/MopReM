%% initialize model
PatchSize = 8;
nmodels = 200;

MiniBatchSize = 1500;
filename = sprintf('GMM_%dx%d_%d_%d',PatchSize,PatchSize,nmodels,MiniBatchSize);

% load images into cell
Images = {};

Files=dir('data');
for k=3:length(Files)
   FileNames=Files(k).name;
   path = strcat('./data/', FileNames);
   Images{k-2} = im2double(rgb2gray(imread(path)));
end

Images

%% learn model from training data
NewGMM = OnlineGMMEM(nmodels,@(N) removeDC(RandPatchesFromImagesCell(N,PatchSize,Images)),1000,MiniBatchSize,filename);

% sort output 
[NewGMM.mixweights,inds] = sort(NewGMM.mixweights,'descend');
NewGMM.covs = NewGMM.covs(:,:,inds);