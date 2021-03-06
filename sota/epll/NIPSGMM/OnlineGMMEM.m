function [GMM,llh] = OnlineGMMEM(GMM0,DataSource,NumIterations,MiniBatchSize,OutputFile,T0,alpha,FirstBatchSize,removeFlatPatches)
% ONLINEGMMEM - Learns a GMM model in an online manner
%
% Inputs:
%
%   GMM0 - if a GMM struct, then the GMM is initialized to the given GMM
%   struct, otherwise this is the number of components to be learned in the
%   mixture
%
%   DataSource - a function handle for a function which gets as a input the
%   number of samples to return and returns them whichever way seems good
%   to you. See "RandPatchesFromImagesCell" for an example of such a
%   function
%
%   NumIterations - how many iterations to learn (in mini batches)
%
%   OutputFile - name of the file to output the intermediate GMM to
%
%   T0 and alpha - learning rate parameters
%
%   FirstBatchSize - size of first mini batch, it helps to make this larger
%   than the mini batches
%
%   removeFlatPatches - if true, we remove any patch with low std before
%   learning
%
% Outputs:
%
%   GMM - the resulting GMM model
%   llh - the log likelihood of each mini batch given through iterations
%
% This code heavily borrows from code by Michael (Mo) Chen (sth4nth@gmail.com)
% which can be downloaded for Matlab's file exchange
%
% Written by Daniel Zoran, 2012 - daniez@cs.huji.ac.il

if isstruct(GMM0)
    GMM = GMM0;
    K = GMM.nmodels;
else
    GMM = [];
    
    K = GMM0;
    GMM.nmodels = K;
    GMM.mixweights = zeros(1,K);
    GMM.covs = zeros(size(DataSource(1),1),size(DataSource(1),1),K);
end

llh = zeros(1,NumIterations);

if ~exist('T0','var')
    T0 = 500;
end
if ~exist('alpha','var')
    alpha = 0.6;
end
if ~exist('FirstBatchSize','var')
    FirstBatchSize = MiniBatchSize*10;
end
if ~exist('removeFlatPatches','var')
    removeFlatPatches = false;
end

% first E step
X = DataSource(FirstBatchSize);
if removeFlatPatches
    inds = find(std(X,1)<0.002);
    X(:,inds) = [];
end
N = size(X,2);
if ~isstruct(GMM0)
    idx = randsample(N,K);
    m = X(:,idx);
    [~,label] = max(bsxfun(@minus,m'*X,sum(m.^2,1)'/2),[],1);
    while K ~= length(unique(label))
        idx = randsample(N,K);
        m = X(:,idx);
        [~,label] = max(bsxfun(@minus,m'*X,sum(m.^2,1)'/2),[],1);
    end
    R = full(sparse(1:N,label,1,N,K,N));
    eta = 1;
else
    % normal E step
    R = zeros(N,K);
    for k = 1:K
        R(:,k) = loggausspdf2(X,GMM.covs(:,:,k))';
    end
    
    R = bsxfun(@plus,R,log(GMM.mixweights));
    T = logsumexp(R,2);
       
    R = bsxfun(@minus,R,T);
    R = exp(R);
    
    eta = (1+T0)^-alpha;
end

for t=1:NumIterations
    % M step
    s = sum(R,1);
    
    % if there are no zero probabilites there, use this mini batch
%     if (all(s>0))
        GMM.mixweights = GMM.mixweights*(1-eta) + eta*s/N;
        for k = 1:K
            sR = sqrt(R(:,k));
            Xo = bsxfun(@times,X(:,sR>0),sR(sR>0)');
            if s(k)>0
                Sig = double((Xo*Xo')/s(k));
                Sig = Sig + 1e-5*eye(size(Xo,1)); 
                % make sure all eigenvalues are larger than 0
                [V,D] = eig(Sig);
                D = diag(D);
                D(D<=0) = 1e-5;
                Sig = V*diag(D)*V';
                Sig = (Sig+Sig')/2;
                GMM.covs(:,:,k) = GMM.covs(:,:,k)*(1-eta) + eta*(Sig);
            end
            
        end

    if t<10
        eta = eta/2;
    else
        eta = (t+T0)^-alpha;
    end
    
    % Get more data!
    if t<10
        X = DataSource(FirstBatchSize);
    else
        X = DataSource(MiniBatchSize);
    end
    if removeFlatPatches
        inds = find(std(X,1)<0.002);
        X(:,inds) = [];
    end
    N = size(X,2);
    
    
    % E step
    R = zeros(N,K);

    % calculate the likelihood on the N-1 leading eigenvectors due to DC
    % removal
    for k = 1:K
        [V,D] = eigs(GMM.covs(:,:,k),size(GMM.covs,1)-1);
        tt = V'*X;
      
        R(:,k) = -((size(D,1))/2)*log(2*pi) - 0.5*sum(log((diag(D)))) - 0.5*sum(tt.*(D\tt),1)';

    end
    
    R = bsxfun(@plus,R,log(GMM.mixweights));
    T = logsumexp(R,2);
    llh(t) = sum(T)/N;
    llh(t) = llh(t)/(size(X,1)-1)/log(2); % loglikelihood
    
    % output
    fprintf('Iteration %d of %d, logL: %.2f %s\n',t,NumIterations,llh(t),OutputFile);
    subplot(1,2,1);
    plot(llh(1:t),'o-'); drawnow;
    subplot(1,2,2);
    plot(sort(GMM.mixweights,'descend')); set(gca,'YLim',[0 1]); drawnow;
    
    
    R = bsxfun(@minus,R,T);
    R = exp(R);
    
      
    if mod(t,2)==0
        save(OutputFile,'GMM','t','NumIterations','eta','MiniBatchSize','llh','alpha','T0');
    end
end