function [l,ll] = GMMLogL(X,GS)
% calculate the log likelihood of patches in X using the GMM model in GS,
% discarding the smallest eigenvector (the DC component).
%
% Daniel Zoran - 2012 daniez@cs.huji.ac.il
ll = zeros(GS.nmodels,size(X,2));

for i=1:GS.nmodels

    [V,D] = eig(GS.covs(:,:,i));
    [~,inds] = sort(diag(D),'descend');
    V = V(:,inds(1:end-1));
    D = diag(D); D = D(inds(1:end-1)); D = diag(D);
    
    t = V'*X;
    ll(i,:) = log(GS.mixweights(i)) - ((size(D,1))/2)*log(2*pi) - 0.5*sum(log(abs(diag(D)))) - 0.5*sum(t.*(D^-1*t),1);
end
l = logsumexp(ll,1);
