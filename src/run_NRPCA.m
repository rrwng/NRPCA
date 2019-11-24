function C = run_NRPCA(X, K, num_run, niter, gauss_noise_level)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function runs NRPCA on the data matrix X and outputs the denoised data matrix with sparse noise removed.
% Input:
%   X: input data matrix with dimension N * P  (N: number of data, P: number of feature)
%   K: number of neighbours
%   num_run: number of iterations of patch updates
% Output:
%   C: a cell containing the denoised data after every iteration
%
% (C) He Lyu, Michigan State University
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   different constants chosen for mnist and swiss roll

N = size(X,1);
L_temp = X;
L = [];
curv = curvature(X);

for i = 1:num_run
    fprintf('------------------------------\n')
    fprintf('Starting round #%d of NRPCA...\n',i);
    [patch, no_copies, A] = update_patch(L_temp, N, K);
    lambda = compute_weight(X ,curv, patch, A, gauss_noise_level);
    L_temp = NRPCA_func(patch, X', lambda, no_copies, niter);
    L = [L',L_temp']';
end
% store the resulting Ls in a cell 
C = mat2cell(L,repmat(N,1,num_run));
end
