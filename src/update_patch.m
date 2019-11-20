function [patch, no_copies, A] = update_patch(L, N, K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% update_patch updates nearest neighbors based
% on current estimated data matrix with sparse noise removal.
% Input:
%   L: data matrix after removing sparse noise
%   N: number of samples
%   K: number of neighbors (including data itself)
% Output:
%   patch: indexes of neighbors, with dimension N * K
%   no_copies: how many times each data point is assigned to a patch
%   A: truncated distance matrix
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
