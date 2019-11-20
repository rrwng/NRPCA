function lambda = compute_weight(X,curv, patch, A, noise_level)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function estimates the regularizing parameters for NRPCA
% Input:
%   X: noisy data matrix with dimension N * P
%   patch: indexes of neighborhoods matrix, with dimension N * (K+1)
%   A: truncated distance matrix
%   noise_level: Gaussian noise level
% Output:
%   lambda: estimated weights
%
% Author: He Lyu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%fprintf('Updating parameters lambda...\n')

[N,P] = size(X);
lambda = zeros(N,1);


K = size(patch,2)-1;
% compute weights for F-norm part
for k = 1 : N
    index = patch(k,2:end);
    E = sum(A(k,index).^4);
    lambda(k) = sqrt(min(K,P))/sqrt(0.25*E*curv(k)^2  +  (K+1)*P * noise_level^2);
end
end

