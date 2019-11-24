clear
clc
setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 1: running NRPCA for High Dimensional (20D) 
% Swissroll dataset.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Swissroll variables declearation
N = 2000;   % total number of samples
P = 3;     % dimension of swiss roll
noise_level = 0.5;    % noise level
sparse_noise_level = 2;    % sparse noise level
[clean_data, noisy_data, cmap] = gen_SwissRoll(N, P, noise_level, sparse_noise_level);
K = 20;     % number of neighbors (including data itself)
num_run = 2; % maximum rounds (T)
niter = 150; % maximum iterations per round 

%% Running NRPCA
% multiple round sparse noise removing with neighbors updated
C = run_NRPCA(noisy_data, K, num_run, niter, noise_level);
L1 = C{1};
L2 = C{2};
% gaussian noise removing with neighbors updated

[L_clean,lambda3] = clean_L(N, K, noisy_data, L2, P, noise_level);

%% Patchwise Robust PCA
L_patch = patch_RPCA(noisy_data, K, noise_level);

%% Visualizing Results using first 3 dimensions
figure()
subplot(2,3,1), scatter3(noisy_data(:,1), noisy_data(:,2), noisy_data(:,3), 10, cmap)
title('Noisy data: $\tilde X$','Interpreter', 'latex','Fontsize',20);
subplot(2,3,2), scatter3(L1(:,1), L1(:,2), L1(:,3), 10, cmap)
title('$\tilde X-\hat S$', 'Interpreter', 'latex','Fontsize',20);
subplot(2,3,3), scatter3(L2(:,1), L2(:,2), L2(:,3), 10, cmap)
title('$\tilde X-\hat S$ with one neighbor update', 'Interpreter', 'latex','Fontsize',20);
subplot(2,3,4), scatter3(L_clean(:,1), L_clean(:,2), L_clean(:,3), 10, cmap)
title('$\hat X$','Interpreter', 'latex','Fontsize',20);
subplot(2,3,5), scatter3(L_patch(:,1), L_patch(:,2), L_patch(:,3), 10, cmap)
title('Patch-wise Roubust PCA','Interpreter', 'latex','Fontsize',20);
subplot(2,3,6), scatter3(clean_data(:,1), clean_data(:,2), clean_data(:,3), 10, cmap)
title('Clean data:$X$','Interpreter', 'latex','Fontsize',20);

%% Embedding via LLE and Laplacian Eigenmap
M = 15;
mapped_clean_lle = compute_mapping(clean_data(:,1:3),'LLE',2,M);
mapped_L1_lle = compute_mapping(L1(:,1:3),'LLE',2,M);
mapped_L2_lle = compute_mapping(L2(:,1:3),'LLE',2,M);
mapped_L_clean_lle = compute_mapping(L_clean(:,1:3),'LLE',2,M);
mapped_noisy_lle = compute_mapping(noisy_data(:,1:3),'LLE',2,M);
mapped_patch_lle = compute_mapping(L_patch(:,1:3),'LLE',2,M);

mapped_clean_lap = compute_mapping(clean_data(:,1:3),'Laplacian',2,M);
mapped_L1_lap = compute_mapping(L1(:,1:3),'Laplacian',2,M);
mapped_L2_lap = compute_mapping(L2(:,1:3),'Laplacian',2,M);
mapped_L_clean_lap = compute_mapping(L_clean(:,1:3),'Laplacian',2,M);
mapped_noisy_lap = compute_mapping(noisy_data(:,1:3),'Laplacian',2,M);
mapped_patch_lap = compute_mapping(L_patch(:,1:3),'Laplacian',2,M);

%% Embedding Results
figure()
subplot(3,2,6), scatter(mapped_clean_lle(:,1), mapped_clean_lle(:,2), 10, cmap), title('Clean data:$X$, LLE','Interpreter', 'latex','Fontsize',15);
subplot(3,2,2), scatter(mapped_L1_lle(:,1), mapped_L1_lle(:,2), 10, cmap), title('$\tilde X-\hat S$, LLE','Interpreter', 'latex','Fontsize',15);
subplot(3,2,3), scatter(mapped_L2_lle(:,1), mapped_L2_lle(:,2), 10, cmap), title('$\tilde X-\hat S$ with one neighbor update, LLE','Interpreter', 'latex','Fontsize',15);
subplot(3,2,4), scatter(mapped_L_clean_lle(:,1), mapped_L_clean_lle(:,2), 10, cmap), title('$\hat X$, LLE','Interpreter', 'latex','Fontsize',15);
subplot(3,2,1), scatter(mapped_noisy_lle(:,1), mapped_noisy_lle(:,2), 10, cmap), title('Noisy data: $\tilde X$, LLE','Interpreter', 'latex','Fontsize',15);
subplot(3,2,5), scatter(mapped_patch_lle(:,1), mapped_patch_lle(:,2), 10, cmap), title('Patch-wise Roubust PCA, LLE','Interpreter', 'latex','Fontsize',15);

figure()
subplot(3,2,6), scatter(mapped_clean_lap(:,1), mapped_clean_lap(:,2), 10, cmap), title('Clean data:$X$, Laplacian','Interpreter', 'latex','Fontsize',15);
subplot(3,2,2), scatter(mapped_L1_lap(:,1), mapped_L1_lap(:,2), 10, cmap), title('$\tilde X-\hat S$, Laplacian','Interpreter', 'latex','Fontsize',15);
subplot(3,2,3), scatter(mapped_L2_lap(:,1), mapped_L2_lap(:,2), 10, cmap), title('$\tilde X-\hat S$ with one neighbor update, Laplacian','Interpreter', 'latex','Fontsize',15);
subplot(3,2,4), scatter(mapped_L_clean_lap(:,1), mapped_L_clean_lap(:,2), 10, cmap), title('$\hat X$, Laplacian','Interpreter', 'latex','Fontsize',15);
subplot(3,2,1), scatter(mapped_noisy_lap(:,1), mapped_noisy_lap(:,2), 10, cmap), title('Noisy data: $\tilde X$, Laplacian','Interpreter', 'latex','Fontsize',15);
subplot(3,2,5), scatter(mapped_patch_lap(:,1), mapped_patch_lap(:,2), 10, cmap), title('Patch-wise Roubust PCA, Laplacian','Interpreter', 'latex','Fontsize',15);

