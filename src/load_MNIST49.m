function [data, cmap] = load_MNIST49(N)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function randomly loads $N$ MNIST digits 4&9.
% Input:
%   N: number of samples per class.
%
% Output
%   data: data matrix of dimension 2N * P.
%   cmap: colormap(labels) for digits.
%
% (C) Ningyu Sha, Michigan State University
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
data = [digit_4(:,perm4(1:N))'; digit_9(:,perm9(1:N))'];

% colormap
cmap = [ones(1,N), 10 * ones(1,N)];

end
