function NMI = nmi(idx, gnd, K, F )
% compute normalized mutual information(NMI)
%input idx: cluster index, N*1 of size
%       gnd: ground-truth index N*1 of size
%  K : number of clusters
%  F: number of ground-truth labels
%output: the normalized mutual infromation
N = size(idx,1);
MI_matrix = zeros(K,F);
for k = 1:K
    for f = 1:F
        temp = sum(gnd(idx==k)==f); %t_{k,f}
        temp1 = sum(idx==k); % t_k
        temp2 = sum(gnd==f); % \hat{t}_f
        if temp == 0
            MI_matrix(k,f) = 0.0;
        else
            MI_matrix(k,f) = temp/N * log2((N*temp/temp1/temp2));
        end
    end
end
MI  = sum(sum(MI_matrix));

H1 = 0.0;
for k=1:K
    temp = sum(idx==k);
    H1 = H1 - temp/N * log2((temp/N));
end

H2 = 0.0;
for f=1:F
    temp = sum(gnd==f);
    H2 = H2 - temp/N * log2((temp/N));
end


NMI = MI/(sqrt(H1*H2));

end