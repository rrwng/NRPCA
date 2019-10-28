function [orig_data, cmap] = load_MNIST49(N)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function load_MNIST49 serves as randomly loading MNIST 4&9 data. 
%
% [orig_data, cmap] = get_mnist49(N)
%
% Input:
%   N: number of samples per class.
%
% Output
%   orig_data: original data matrix of dimension 2N * P.
%   cmap: colormap(labels) for digits.
%
% Author: He Lyu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Loading MNIST data...\n')

temp = load('data/digit_4.mat');
digit_4 = temp.digit_4;
temp = load('data/digit_9.mat');
digit_9 = temp.digit_9;
n4 = size(digit_4,2);
n9 = size(digit_9,2);

if N>n4 || N>n9
    error('N too large, choose a smaller N.')
end

% randomly choose N samples within each class
perm4 = randperm(n4);
perm9 = randperm(n9);
orig_data = [digit_4(:,perm4(1:N))'; digit_9(:,perm9(1:N))'];

% colormap
cmap = [ones(1,N), 10 * ones(1,N)];

end
