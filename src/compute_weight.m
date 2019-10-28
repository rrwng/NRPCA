function lambda = compute_weight(X,curv, patch, A, noise_level)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function get_weight estimates the regularizing parameter a following
% (eq.(12)).
%
% lambda = get_weight(orig_data,patch,D,D,K,P,d,r,const1,const2,noise_level,no_copies)
% Input:
%   orig_data: input raw data matrix with dimension N * P
%   patch: indexes of neighborhoods matrix, with dimension N * (K+1)
%   A: truncated distance matrix
%   D: full pairwise distance matrix
%   K: number of neighbors
%   P: original data dimension
%   d,r: constants for finding neighborhood samples within range [d,r]
%   const1, const2: a larger constant, for finding samples in computing
%       geodesic distances
%   noise_level: Gaussian noise level
%   no_copies: sample repetition in patch matrix
% Output:
%   lambda: estimated weights for F-norm part
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

