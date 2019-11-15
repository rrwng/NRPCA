function C = run_NRPCA(X, K, num_run, niter, gauss_noise_level)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function run_NRPCA runs the proposed NRPCA method on orig_data for
% num_run times, and output the estimation of denoised data matrix.
%
% L = run_NRPCA(orig_data, K, num_run)
% Input:
%   orig_data: input data matrix with dimension N * P
%   K:
%   num_run: number of total running iterations
% Output:
%   L1: first denoised estimation
%   L2: last denoised estimation
%   lambda1: lambda estimated from noisy data
%   lambda2: lambda estimated from data matrix with sparse noise removal
%
% Author: He Lyu
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
    %curv = curvature(L_temp);
    lambda = compute_weight(X ,curv, patch, A, gauss_noise_level);
    L_temp = NRPCA_func(patch, X', lambda, no_copies, niter);
    L = [L',L_temp']';
end

C = mat2cell(L,repmat(N,1,num_run));
end