function lambda = optimal_hard_thresh(P,K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function optimal_hard_thresh computes the optimal hard thresholds for
% singular values. Following eq.(10)&(11) in reference below.
%   Reference: The Optimal Hard Threshhold for Singular Values is 4/sqrt(3)
%   Authors: Matan Gavish, David L. Donoho
%
% Author: He Lyu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

beta = min(P/K, K/P);
lambda = sqrt(2*(beta+1)+8*beta/((beta+1)+sqrt(beta^2+14*beta+1)));

end