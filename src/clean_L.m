function [clean_L,lambda] = clean_L(N, K, X, L2, P, noise_level, curv)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function denoised_L computes the final estimation clean data matrix,
% by removing gaussian noise. Following eq.(13)&(15):
%   L = (sum_{i=1}^n lambda_i * P^i * P^i')^{-1}*(sum_1^n lambda_i * P^i' * L^i)
%
% clean_L = clean_data(N, K, orig_data, L2, P, lambda)
% Input:
%   N: number of data points
%   K: number of neighbors (including data itself)
%   patch: indexes of neighborhoods matrix, with dimension N * (K+1)
%   orig_data: original data matrix
%   L2: estimated data matrix obtained by NRPCA with T=2
%   P: original dimensionality of data
%   lambda: regularizing parameter before F-norm in objective function
%   no_copies: a vector that counts the number of local patches the i-th point belongs to  
% Output:
%   clean_L: final estimation \hat X of clean data matrix
%
% Author: He Lyu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('-------------------------------\n')
fprintf('Estimating clean data matrix...\n')

[patch, no_copies, A] = update_patch(L2, N, K);
lambda = compute_weight(L2,curv, patch, A, noise_level);
GB1 = zeros(N, N * K);
patch = patch';

% construct (restriction - centering) operator
for i = 1: N
    GB1(patch(:, i), ((i - 1) * K + 1) : (i * K)) = eye(K, K)-1/K*ones(K,K);
end

% construct centering operator
GB2 = zeros(N, N * K);
for i = 1: N
    GB2(patch(:, i), ((i - 1) * K + 1) : (i * K)) = 1/K*ones(K,K);
end

% current estimation of sparse noise
S_recovered = X - L2;
L_all = zeros(P, K*N);
A = zeros(size(diag(no_copies)));
B=zeros(P,N);
for i = 1 : N
    GB_i = GB1(:, ((i-1) * K + 1) : (i * K));
    GB2_i = GB2(:, ((i-1) * K + 1) : (i * K));
    S_i = S_recovered' * GB_i;
    X_i = X' * GB_i;
    [U, Sigma_i, V] = svd(X_i - S_i, 'econ');

    % optimal thresholding for singular value
    scale = sqrt(max(K,P))*0.5;
    Sigma_i(diag(Sigma_i) / scale < optimal_hard_thresh(P, K),:)=0;
    
    L_i = U * Sigma_i * V';
    % estimation of local data matrix L by fixing S
    L_i = L_i + (X'-S_recovered')*GB2_i;
    L_all(:, ((i-1) * K + 1) : (i * K)) = L_i;
    A = A + lambda(i) * (GB_i + GB2_i) * (GB_i + GB2_i)';
    B = B + lambda(i) * L_i * (GB_i + GB2_i)';
end

clean_L = B / A;
clean_L = clean_L';

end
