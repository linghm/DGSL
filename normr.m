function Y = normr(X)
sumX = sqrt(sum(X.^2,2));
temp = repmat(sumX,1,size(X,2)) + eps;
Y = X./temp;