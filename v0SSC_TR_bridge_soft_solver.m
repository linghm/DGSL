function [A,Z,F,err] = v0SSC_TR_bridge_soft_solver(X, W_SC, W_cannotlink,W_mustlink, para,display,debug) 

alpha1 = para.alpha1;
alpha2 = para.alpha2;
lambda = para.lambda;
lambda_Z = para.lambda_Z;
lambda_M = para.lambda_M;
k = para.k;
maxiter = para.maxiter;
tol = para.tol;

n = size(X,2);
XtX = X'*X;
I = eye(n);
invXtXI = I/(XtX+lambda*I);

 
alpha1overlambda = alpha1/lambda;
Z = zeros(n,n);
F = zeros(n,k);
A = zeros(n,n);
iter = 0;
while iter < maxiter
    iter = iter + 1;        
    if  true
        %update F
        CKSym = 0.5 * BuildAdjacency(Z); 
        W =  alpha1* CKSym +  alpha2 * (W_SC + lambda_M *  W_mustlink);
        L_mustlink = diag(sum(W,2)) - W;
        
        num_cannotlink = nnz(W_cannotlink);
        L_cannotlink = diag(sum(W_cannotlink,2)) - W_cannotlink;
        L_cannotlink = 1/num_cannotlink * L_cannotlink;
        
       
        temp = sum(W,2);
        index = find(temp>0);
        temp(index) = temp(index).^-0.5;                
        D_inv = sparse(1:n,1:n,temp,n,n);
        L_mustlink = D_inv * L_mustlink *D_inv;      
        

        [F,~] = trace_ratio_optim(L_cannotlink,L_mustlink,k,20);
        % normalized rows of F
        Fk = F;
        F = normr(real(F));
    end
    %udpate A
    Ak = A;
    A = invXtXI * (XtX  + lambda *Z);
    
    % update Z
    Zk = Z;
    temp = Fk' * L_cannotlink *Fk; 
    
    tr = 2 * trace(temp);
    threshold = alpha1overlambda * 0.5 * squareform(pdist(F,'squaredeuclidean'))./tr + lambda_Z/lambda * ones(n,n);
    
    Z = sign(A) .* max(0, abs(A) - threshold);
    Z = Z-diag(diag(Z));
    
    

    diffZ = max(max(abs(Z-Zk)));
    diffA = max(max(abs(A-Ak))); 
  
    stopC = max([diffZ,diffA]);
    if display && (iter==1 || mod(iter,10)==0 || stopC<tol)
        disp(['iter ' num2str(iter) ',lambda=' num2str(lambda,'%2.1e') ...
                ',nnzZ=' num2str(nnz(Z))   ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if debug
        err.diffZs(iter) = diffZ;
        err.diffAs(iter) = diffA;
        err.stopCs(iter) = stopC;
        [grps,~] = kmeans(F,k,'maxiter',1000,'replicates',20,'EmptyAction','singleton');
        err.missrateF(iter) = 1 - munkres_acc_cal(grps,Y,k);
        err.NMIF(iter) = nmi(grps, Y, k,k);
    end
    if stopC < tol 
        fprintf('convergence after iteration %d\n',iter);
        break;   
    end
end


