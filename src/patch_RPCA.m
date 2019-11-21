function L_patch = patch_RPCA(X, K, noise_level)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function estimates the clean data matrix as L_patch via
% patch-wise robust PCA.
%
% Input:
%   orig_data: input data matrix with dimension N * P
%   K: number of neighbors (including itself)
%   gauss_noise_level: Gaussian noise level
% Output:
%   L_patch: estimation of clean data matrix
%
% (C) Ningyu Sha, Michigan State University
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('----------------------------------\n')
fprintf('Start solving patchwise RPCA...\n')

global mu lambda

curv = curvature(X);
[N, P] = size(X);
[patch, ~, A] = update_patch(X, N, K);
lambda = compute_weight(X,curv, patch, A, noise_level);
mu = 1/sqrt(max(K,P));
B = zeros(N, N * K);

patch = patch';
X = X';
for i = 1: N
    B(patch(:, i), ((i - 1) * K + 1) : (i * K)) = eye(K, K)-(1/K) * ones(K, K);
end

%% Patchwise Robust PCA solver variables
obj          = solver_patchRPCA;

obj.gradf      = @(Xi, Si, entry) feval(@f_grad, Xi, Si, entry);
obj.proxg      = @(Si, t) feval(@g_prox, Si, t);

obj.para.Si    = zeros(P, K); % P * N
obj.para.iter  = 200;
obj.para.t     = 1/3; 

%% Estimating sparse noise
% estimating sparse noise per patch
S_final = zeros(P, N);
S_total = zeros(P, N * K);
for i = 1 : N
    obj.para.Xi = X(:, patch(:, i));
    obj.para.entry = i;
    [S_i, ~] = obj.loss_minimizer();  
    S_total(:, ((i - 1) * K + 1) : (i * K)) = S_i;
end

% averaging local patch to get global estimation 
for i = 1 : N
    [rows, cols] = find(patch == i);
    S_col = zeros(P, 1);
    for j = 1 : length(rows)
        S_col = S_col + S_total(:, (cols(j) - 1) * K + rows(j));
    end
    S_final(:, i) = S_col/length(rows);
end

% estimation of clean data
L_patch = (X - S_final)';

end

%% calculate gradient of f
function y = f_grad(Xi, Si, entry)
global lambda
[U, Sigma, V] = svd(Xi - Si, 'econ');
Sigmahat = max(Sigma - 1/(2*lambda(entry)), 0);
Li = U * Sigmahat * V';
y = 2 * lambda(entry) * (Li + Si - Xi);   
end

%% calculate proximal of g
function y = g_prox(Si, t)
    global mu
    y = max(0, Si -  t * mu) - max(0, - Si - t * mu);
end
