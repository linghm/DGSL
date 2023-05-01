rng('default')
debug = false;
totalrun= 20;

load ./dataset/ORL.mat
% load ./dataset/Yale_32x32.mat
% X = fea;
% Y = gnd;
k = length(unique(Y));
n = size(X,1);
mode = 1;
% num_mustlink =100;
% num_cannotlink = 3 * num_mustlink;
f = 2 * k;
para.k = k;
display = true;
para.tol = 1e-2;  % convergence tolerance, default 1e-2
para.maxiter = 50;  % maximum number of iterations, default 50
para.lambda = 100;    % default 100
para.alpha2overalpha1 = 0.2; 
para.thresholds = 0.01; % corresponds to \tau in the paper
para.thr_Zs = 0.00; 
% para.lambda_Z = para.thr_Z * para.lambda;    
para.lambda_Ms = [10]; 

tic
X = X';
gnd = Y;
% simple scalar for ORL Yale Yale B Umist
X = X./255.0;

Knearest =7;
Knearest_sigma = 5;
W_SC = weight_calv2(X',Knearest,Knearest_sigma); 
W_SC = (W_SC + W_SC') * 0.5;


for outer_iter = 1: length(para.lambda_Ms)
    para.lambda_M = para.lambda_Ms(outer_iter);
    for middle_iter  = 1 : length(para.thr_Zs)
        para.thr_Z = para.thr_Zs(middle_iter);
        para.lambda_Z = para.thr_Z * para.lambda;
        for iter = 1: length(para.thresholds)
            para.threshold = para.thresholds(iter);
            for runid = 1: totalrun
                rng(1000* runid)

                switch(mode)
                    case 1
                        disp('construct pairwise constraints by partial labels')
                        idx_fidelity = zeros(f,1);
                        h = f/k;
                        for t = 1:k
                            index = find(gnd==t);
                            random_sampler = randperm(length(index));
                            idx_fidelity(((t-1)*h+1):(t*h),:) = index(random_sampler(1:h));
                        end
                        fidelity = [idx_fidelity, gnd(idx_fidelity)];
                        [W_cannotlink,W_mustlink] = construct_link(W_SC,fidelity,k,1);
                        num_cannotlink = nnz(W_cannotlink);
                    case 2
                        % construct random pairwise constraints
                        [W_cannotlink,W_mustlink] = construct_pairwise_link(W_SC, Y,k, num_mustlink, num_cannotlink,1,mode);
                    case 3
                        % construct key pairwise constraints
                        [W_cannotlink,W_mustlink] = construct_pairwise_link(W_SC, Y,k, num_mustlink, num_cannotlink,1,mode);
                    otherwise
                        error('wrong mode number')
                end

                temp = W_SC + para.lambda_M * W_mustlink;
                D_inv = diag(1./sqrt(sum(temp,2)));
                L_mustlink = diag(sum(temp,2)) - temp;
                L_cannotlink = diag(sum(W_cannotlink,2)) - W_cannotlink;
                L_mustlink = D_inv * L_mustlink * D_inv;
                L_cannotlink = 1/num_cannotlink * L_cannotlink; 
                
                [Vs,maximum] = trace_ratio_optim( L_cannotlink,L_mustlink,k,20);
                temp = Vs' * L_cannotlink *Vs;
                tr = 2 * trace(temp);
                % define alpha1 and alpha2
                para.alpha1 = para.threshold * para.lambda * tr;
                para.alpha2 = para.alpha2overalpha1 * para.alpha1;

                [Z,B,F] = v0SSC_TR_bridge_soft_solver(X,W_SC, W_cannotlink,W_mustlink, para, display,debug); 
                
                % use B
                CKSym = BuildAdjacency(B);
                grps = SpectralClustering(CKSym,k);
                ACCB(runid) = munkres_acc_cal(grps,Y,k);
                NMIB(runid) = nmi(grps, Y, k,k);
                % use Z
                CKSym = BuildAdjacency(Z);
                grps = SpectralClustering(CKSym,k);
                ACCZ(runid) = munkres_acc_cal(grps,Y,k);
                NMIZ(runid) = nmi(grps, Y, k,k);
                % use F
                F = normr(F);
                [grps,~] = kmeans(F,k,'maxiter',1000,'replicates',20,'EmptyAction','singleton');
                ACCF(runid) = munkres_acc_cal(grps,Y,k);
                NMIF(runid) = nmi(grps, Y, k,k);
                fprintf('\t runid: %.1f  used B: %.3f, use Z: %.3f   use F: %.3f  \n',runid, ACCB(runid),ACCZ(runid), ACCF(runid));
            end
            fprintf('\t avgB: %.4f, medianB: %.4f  stdB: %.4f \n',mean(ACCB),median(ACCB), std(ACCB));
            fprintf('\t avgZ: %.4f, medianZ: %.4f  stdZ: %.4f \n',mean(ACCZ),median(ACCZ), std(ACCZ));
            fprintf('\t avgF: %.4f, medianF: %.4f  stdF: %.4f \n',mean(ACCF),median(ACCF), std(ACCF));
            toc
            if mode == 1
                output1 = [ 'mode=' num2str(mode,'%.1f'),',f=' num2str(f,'%.1f')];
            elseif mode == 2 || mode == 3
                output1 = [ 'mode=' num2str(mode,'%.1f'),'num_mustlink=' num2str(num_mustlink,'%.1f'), ',num_cannotlink=' num2str(num_cannotlink,'%.1f')];
            end
            output2 = ['dataset:ORL' , ',totalrun=' num2str(totalrun),  ',nCluster=' num2str(k),',lambda=' num2str(para.lambda), ...
                      ',thr=' num2str(para.threshold),  ',thr_Z=' num2str(para.thr_Z),',lambda_Z=' num2str(para.lambda_Z), ...
                      ',alpha1=' num2str(para.alpha1,'%.4f'),  ',alpha2=' num2str(para.alpha2,'%.4f'), ',alpha2overalpha1=' num2str(para.alpha2overalpha1,'%.4f'),...
                    ',lambda_M=' num2str(para.lambda_M), ',Knearest_neighbor=(' num2str(Knearest) ',' num2str(Knearest_sigma) ')'...           
                      ',tol=' num2str(para.tol,'%.4f'), ',maxiter=' num2str(para.maxiter)
                      ];
            output3 = ['ACC:',...
                      ',avgrateB=' num2str(mean(ACCB),'%.4f'),',medianB=' num2str(median(ACCB),'%.4f'),',stdB=' num2str(std(ACCB),'%.4f'), '\n', ...
                      ',avgrateZ=' num2str(mean(ACCZ),'%.4f'),',medianZ=' num2str(median(ACCZ),'%.4f'),',stdZ=' num2str(std(ACCZ),'%.4f'), '\n',...
                      ',avgrateF=' num2str(mean(ACCF),'%.4f'),',medianF=' num2str(median(ACCF),'%.4f'),',stdF=' num2str(std(ACCF),'%.4f'), '\n'
                      ];
            output4 = ['NMI:',...
                      ',avgrateB=' num2str(mean(NMIB),'%.4f'),',medianB=' num2str(median(NMIB),'%.4f'),',stdB=' num2str(std(NMIB),'%.4f'), '\n', ...
                      ',avgrateZ=' num2str(mean(NMIZ),'%.4f'),',medianZ=' num2str(median(NMIZ),'%.4f'),',stdZ=' num2str(std(NMIZ),'%.4f'), '\n',...
                      ',avgrateF=' num2str(mean(NMIF),'%.4f'),',medianF=' num2str(median(NMIF),'%.4f'),',stdF=' num2str(std(NMIF),'%.4f'),'\n'
                      ];        
            fid = fopen('./results/ORL_results.txt','a');
            fprintf(fid, '%s\n', output1);
            fprintf(fid, '%s\n', output2);
            fprintf(fid, '%s\n', output3);
            fprintf(fid, '%s\n', output4);
            fclose(fid);
        end
    end
end