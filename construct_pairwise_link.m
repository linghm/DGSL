function [W_cannotlink,W_mustlink] = construct_pairwise_link(W, gnd,k, num_mustlink, num_cannotlink,lambda,mode)
% the labels in gnd is sorted in a assending order
n = size(W,1);
classlen = zeros(1,k);
for t = 1:k
    classlen(1,t) = sum(gnd==t);
end
classlen = [0,classlen];
classlen = cumsum(classlen);

% choose the pairwise constraints randomly
switch(mode)
    case 2
        disp('use the random pairwise constrains')
        W = ones(size(W));
        for t = 1:k
            W((classlen(t)+1):classlen(t+1),(classlen(t)+1):classlen(t+1) ) = 0;
        end
    case 3 
        disp('use the key pairwise constraints')
    otherwise
        error('wrong mode number')
end


W = (W+W')*0.5;
W_block = zeros(n,n);
W_antiblock = W;
for t = 1:k
    temp = W((classlen(t)+1):classlen(t+1),(classlen(t)+1):classlen(t+1) );
    temp2 = temp < 1e-10;
    W_block((classlen(t)+1):classlen(t+1),(classlen(t)+1):classlen(t+1) ) = temp2;
    W_antiblock((classlen(t)+1):classlen(t+1),(classlen(t)+1):classlen(t+1)) = 0;
end

[row,col] = find(W_antiblock > 1e-10);
index = randperm(length(row));
W_cannotlink = sparse(row(index(1:num_cannotlink)),col(index(1:num_cannotlink)),ones(1,num_cannotlink), n,n);
W_cannotlink = W_cannotlink + W_cannotlink';
W_cannotlink = W_cannotlink > 0.5;

[row,col] = find(W_block>1e-10);
index = randperm(length(row));
W_mustlink = sparse(row(index(1:num_mustlink)),col(index(1:num_mustlink)),  ones(1,num_mustlink), n,n);
W_mustlink = W_mustlink + W_mustlink';
W_mustlink = W_mustlink > 0.5;
W_mustlink = W_mustlink * lambda;
end