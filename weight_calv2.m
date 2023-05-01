function y=weight_calv2(X, num_neighbors,rank_neighbor)
% X: size n * m, n is the number of samples
% num_neighbors: total number of neighbors for each sample
% rank_neighbor: the rank_neighbor-th nearest distance as sigma

[n,~]=size(X);
num_s =num_neighbors;
dist = squareform(pdist(X));

[dist,idx] = sort(dist,2); % sort each row of dist in ascending order and return the index idx
dist = dist(:,1:num_neighbors);
idx = idx(:,1:num_neighbors);

sigma=sparse(1:n,1:n, 1./max(dist(:,rank_neighbor),1e-2),n,n);  

id_row=repmat([1:n]',1,num_s);
id_col=double(idx);
w=exp(-(sigma * dist).^2);
y=sparse(id_row,id_col,w,n,n);

% set the diagonal elements zero
y = y - diag(diag(y));
    