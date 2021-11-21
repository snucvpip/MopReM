function L = E_KL(y,x,Model)
exact =0;
L=0;
n=1;
if exact
    
  wy = Model.w * y / 2;

     logiC = Model.b*(2*diag(wy(:,n)))*Model.b';
    C = expm(logiC);
    L = L - .5*trace(logiC);
    L = L - .5 * x(:,n)' * (C \ x(:,n));
    L = L - (size(x,1)/2)*log(2*pi);
else
  % this uses series approximation to expm (much faster)
  BB = Model.btb;
  
  wy = Model.w * y / 2;

  C0 = (Model.b' * x).* wy;
  C1 = (BB * C0).*wy;  C2 = (BB * C1).*wy;
  C3 = (BB * C2).*wy;  C4 = (BB * C3).*wy;
  C5 = (BB * C4).*wy;

  C =  x +  Model.b * (-C0 + C1 / 2 - C2/6 + C3/24 - C4/120 + C5/720);

  L = -.5 * (sum(C.^2,1)) - .5 * (sum(2*wy,1));
  
  L = L - (size(x,1)/2)*log(2*pi);
end 

% lalpacian prior on y
L = L + sum(logggpdf(y,0,1,1));
% gaussian prior on y
%L = L - sum(sum(Data.y.^2));

L = -L;

% this is my version - the energy function is right, something is a tiny
% bit wrong with the derivative, not sure what.
% function [f,df] = ff(y,u,B,W)
% wy = W*y;
% logC = B*diag(wy)*B';
% C = expm(logC);
% 
% % f = -0.5*trace(logC) - 0.5*(u'*C*u) + sum(logggpdf(y,0,1,1)) - (size(u,1)/2)*log(2*pi) ;
% f = loggausspdf(u,zeros(size(u)),C) + sum(logggpdf(y,0,1,1));
% df = zeros(size(y));
% for l=1:size(y,1)
%     df(l) = -0.5*sum(W(:,l)) + 0.5*(u'*(C\B*diag(W(:,l))*B')*u) + dxlogggpdf(y(l),0,1,1);
% end