function [patch, no_copies, A] = update_patch(L, N, K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function update_patch updates nearest neighbors, distance matrices based
% on current estimated data matrix with sparse noise removal. We can use
% the new patches to run proposed NRPCA using updated patch information.
%
% [patch, no_copies, A, D, d] = update_patch(L, N, K)
% Input:
%   L: data matrix after removing sparse noise
%   N: number of samples
%   K: number of neighbors (including data itself)
% Output:
%   patch: indexes of neighbors, with dimension N * K
%   no_copies: weights to compute regularizing parameter mu, by taking
%       sum beta*||S^i||_1 out of parenthesis to be mu * ||S||_1
%   A: truncated distance matrix
%   D: Full distance matrix
%   d: constant to compute geodesic distance
%
% Author: He Lyu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Updating local patches...\n')

% update distance matrix
[A,D] = distance_matrix(L, K-1);
d = mean(max(A,[],2));
patch = zeros(N, K);

% update neighbors
for i = 1 : N
    patch(i,2:end) = find(A(i,:)~=0);
    patch(i,1) = i;
end

% update weights for beta
no_copies = zeros(N,1);
for i = 1 : N
    no_copies(i) = size(find(patch(:)==i),1);
end

end
