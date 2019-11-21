function [A, D] = distance_matrix(data, K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the full and truncated pairwise
% Euclidean distances between data points in columns.
%   
% [A, D]=distance_matrix(data,K)
% Input:
%   data: input data matrix, with dimension P * N
%   K: number of neighbors (not including the point itself)
% Output:
%   A: truncated distance matrix, keeping only (k+1) smallest distances in each row (including itself)
%   D: full pairwise distance matrix, with dimension N * N
%
% (C) He Lyu, Michigan State University
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = size(data, 1);
M = repmat(sum(data .* data, 2), 1, N);
D = sqrt(abs(M + M' - 2 * (data * data')));
D = D - diag(diag(D));

D1 = zeros(N);
for i =  1:N
    [~, index] = sort(D(i, :));
    D1(i, index(1:K+1)) = 1;
end

A = D .* D1;
end
