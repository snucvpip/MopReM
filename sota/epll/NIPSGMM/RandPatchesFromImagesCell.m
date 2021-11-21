function [X,ns,xs,ys] = RandPatchesFromImagesCell(MiniBatchSize,PatchSize,Images,ns,xs,ys)
% return a set of MiniBatchSize patches sized PatchSizexPatchSize from the
% 3D array Images (WxHxN) all assumed to be of the same size

[N] = length(Images);

X = zeros(PatchSize,PatchSize,MiniBatchSize);
shouldSample = false;
if ~exist('ns','var')
    ns = randsample(N,MiniBatchSize,true);
    xs = [];
    ys = [];
    shouldSample = true;
end
for n=sort(unique(ns)')
    inds = find(ns==n);
    [H,W] = size(Images{n});
    if shouldSample
        ys(inds) = randsample(H-PatchSize+1,length(inds),true);
        xs(inds) = randsample(W-PatchSize+1,length(inds),true);
    end
    image_inds = sub2ind([H W],ys(inds),xs(inds));
    for y=0:PatchSize-1
        for x=0:PatchSize-1
            X(y+1,x+1,inds) = Images{n}(image_inds + y + x*H);
        end
    end
end

X = reshape(X,[PatchSize^2 MiniBatchSize]);

% if strcmp(class(Images{1}),'uint8')
%     X = single(X/255);
% end