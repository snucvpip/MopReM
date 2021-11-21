function dy = dEdy_KL(y,x,Model)
% use Taylor series expansion for the exponential
BB = Model.btb;  wy = -Model.w*y;


C0 = Model.b'*x;  
C1 = BB * (C0 .* wy); 
C2 = BB * (C1 .* wy);  
C3 = BB * (C2 .* wy); 
C4 = BB * (C3 .* wy);  
C5 = BB * (C4 .* wy);

C = C0.*(C0 + C1 + C2/3 + C3/12 + C4/60  + C5/360) + ...
    C1.*(          C1/6 + C2/12 + C3/60  + C4/360 + C5/2520) + ...
    C2.*(                         C2/120 + C3/360 + C4/2520);

% log posterior = log likelihood + log prior = log p(x|B,w,y) + log p(y)
% (likelihood) d p(x|B,w,y)/dy
dy = .5 * Model.w' * (C - 1);

% sparse (Laplacian) prior on y, d p(y)/dy
dy = dy + dxlogggpdf(y,0,1,1);
dy = -dy;