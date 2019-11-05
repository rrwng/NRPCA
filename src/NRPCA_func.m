function L = NRPCA_func(patch, data, lambda, no_copies, niter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The function NRPCA_func computes the denoised result 'L' of the
% original data matrix by solving the optimization problem:
%   S = argmin sum_{i=1}^n (lambda_i * || X^i - L^i - S^i ||_F^2) + ||C(L^i)||_* + mu * || S^i ||_1)
%
% L = NRPCA_func(patch, data, lambda, no_copies, niter)
% Input:
%   patch = (K+1) * N matrix, contains the indexes of K+1 nearest
%       neighbors including itself as i^th row for data point x_i.
%   X = P * N input data matrix
%   lambda: regularizing parameter for the F-norm
%   no_copies: frequen
%   niter: number of iterations
% Output:
%   L: denoised data matrix (sparse noise removed)
%
% Author: Ningyu Sha
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Start solving minimization problem...\n')

if size(patch,1)>size(patch,2)
    patch = patch';
end

%%  Initialize Global Variables
%   K: number of neighborhoods including itself
%   N: number of samples
%   P: original data dimension

global mu lambda_copy GB data_copy N P K
K = size(patch, 1);
N = size(data, 2);
P = size(data, 1);
data_copy = data;
lambda_copy = lambda;

mu = (no_copies/sqrt(max(K,P)));
%% global variables GB

%centered

GB = zeros(N, N * K);


for i = 1: N
    GB(patch(:, i), ((i - 1) * K + 1) : (i * K)) = eye(K, K)-(1/K) * ones(K, K);
end


%% start using solver_NRPCA class
obj          = solver_NRPCA;

obj.funf       = @(S) feval(@f_val, S);
obj.fung       = @(S) feval(@g_val, S);
obj.gradf      = @(S) feval(@f_grad, S);
obj.proxg      = @(S, t) feval(@g_prox, S, t);

obj.para.S     = zeros(size(data)); % P * N
obj.para.iter  = niter;
obj.para.t     = 0.005;
obj.para.X     = data;

%% call main function
L = obj.loss_minimizer();
L = L';

end

%% Calculate gradient of f part
function y = f_grad(S)

global N P GB data_copy K lambda_copy 
y = zeros(P, N);

for i = 1 : N
    GB_i = GB(:, ((i-1) * K + 1) : (i * K));
    S_i = S * GB_i;
    X_i = data_copy * GB_i;
    [U, Sigma_i, V] = svd(X_i - S_i, 'econ');
    Sigmahat_i = max(Sigma_i - 1/(2*lambda_copy(i)), 0);
    % estimation of local data matrix L^i by fixing S
    L_i = U * Sigmahat_i * V';
    temp = 2 * lambda_copy(i) * (L_i + S_i - X_i) * GB_i';
    y = y + temp;
end

end


%% calculate loss function of f part
function y = f_val(S) % S : P * N

y = 0;
global N K GB data_copy lambda_copy 

for i = 1 : N
    GB_i = GB(:, ((i-1) * K + 1) : (i * K));
    S_i = S * GB_i;
    X_i = data_copy * GB_i;
    [U, Sigma_i, V] = svd(X_i - S_i, 'econ');
    Sigmahat_i = max(Sigma_i - 1/(2*lambda_copy(i)), 0);
    % estimation of local data matrix L^i by fixing S
    L_i = U * Sigmahat_i * V';
    temp = lambda_copy(i) * norm(X_i - L_i - S_i, 'fro')^2 + sum(svd(L_i));
    y = y + temp;
end

end

%% calculate proximal of g
function y = g_prox(S, t)

global mu P

y = max(0, S -  t * ones(P, 1) * mu') - max(0, - S - t* ones(P, 1) * mu');

end


%% calculate loss function of g part
function y = g_val(S)

y = 0;
global mu N

for i = 1 : N
    y = y + mu(i) * norm(S(:,i), 1);
end

end
