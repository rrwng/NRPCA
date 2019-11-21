function [clean_data, noisy_data, cmap] = gen_SwissRoll(N, P, gauss_noise_level, sparse_noise_level)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function generates high dimensional swiss roll data.
%
% [orig_data, cmap] = load_SwissRoll(N, P, gauss_noise_level, sparse_noise_level)
%
% Input:
%   N: number of samples
%   P: dimension of the high-dimensional Swiss roll data
%   gauss_noise_level: Gaussian noise level
%   sparse_noise_level: sparse noise level
%
% Output
%   clean_data: clean data matrix
%   noisy_data: noisy data matrix of dimension N * P.
%   cmap: colormap(labels) for swiss roll.
%
% (C): He Lyu, Michigan State University
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Generating Swiss Roll data...\n')
 
% Generate swiss roll
t = sort(4 * pi * sqrt(rand(N, 1)));    % parameter for spiral
x = (t + 1).* cos(t) + 0 * rand(N, 1) + 30;
y = (t + 1).* sin(t) + 0 * rand(N, 1) + 30;
z = 8 * pi * rand(N,1); % random heights, second parameter
clean_data = zeros(N, P);
clean_data(:, 1:3) = [x, y, z];
for dim = 4: P
    clean_data(:, dim) = t .* sin(dim * t/21);
end

% Generate colormap for Swiss Roll
t_max = max(t)^2 + 1;
cmap = ind2rgb(uint8((256/t_max) * (t.^2+1)), jet(256));
cmap = squeeze(cmap);

no_outlier = ceil(N * P * 0.015);  % number of outliers (sparse noise exists)
sparse_loc = randsample(1 : N, no_outlier);
sparse_dim = randsample([1:2, 4:P], no_outlier,true);
    

sparse_noise = zeros(N, P);
for i = 1:no_outlier
    sparse_noise(sparse_loc(i), sparse_dim(i)) = normrnd(sparse_noise_level, .3)*(-1)^(binornd(1,0.5));
end
gaussian_noise = normrnd(0, gauss_noise_level, [N, P]);
noise = sparse_noise + gaussian_noise;

noisy_data = clean_data + noise;

end    
