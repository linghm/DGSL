function [Vs,rho] = trace_ratio_optim(A,B,k,max_iter)
[Vs,ds] = eigs(A + speye(size(A,1)), k, 'LA');  
% [Vs, ds] = eig(full(A));
% ds = diag(ds);
% [~, ind] = sort(ds,'descend');    
% Vs = Vs(:,ind(1:k));
rho = trace(Vs'* A * Vs) / trace(Vs' * B * Vs);
iter = 0;
while iter == 0 ||  abs(rho - rho_before) > 1e-8 %default 1e-8
    [Vs,ds] = eigs(A- rho* B + 1000 *speye(size(A)), k, 'LA');
%     temp = full(A-rho*B);
%     [Vs, ds] = eig(temp);
%     ds = diag(ds);
%     [~, ind] = sort(ds,'descend');    
%     Vs = Vs(:,ind(1:k)); 
    rho_before = rho;
    rho = trace(Vs'* A * Vs) / trace(Vs' * B * Vs);
    iter = iter +1;
    if iter > max_iter
        break
    end
end
if iter >= max_iter
    fprintf('the newton iteration fails to converge in trace ratio')
end
end