function [W_cannotlink,W_mustlink] = construct_link(W,fidelity,k0, lambda)
n = size(W,1);
W_mustlink = zeros(n,n);

labeled_class = unique(fidelity(:,2));
for t = 1: k0
    index = fidelity(fidelity(:,2)==labeled_class(t),1);   % S_t
    W_mustlink(index,index) = lambda *  ones(length(index),length(index));
end
W_mustlink = W_mustlink - diag(diag(W_mustlink));
row_index = [];
col_index = [];
for jj = 1:k0
    index1 = fidelity(fidelity(:,2)==labeled_class(jj),1)';
    for tt = 1:k0
        index2 = fidelity(fidelity(:,2)==labeled_class(tt),1)';
        if jj~=tt
            temp1 = reshape(repmat(index1,length(index2),1),1, length(index1)*length(index2));
            temp2 = reshape(repmat(index2,length(index1),1)',1, length(index1)*length(index2));
            row_index = [row_index, temp1];
            col_index = [col_index,temp2];
        end
    end
end
W_cannotlink = sparse(row_index,col_index,ones(1,length(row_index)), n,n);
W_cannotlink = W_cannotlink>0.5;
end