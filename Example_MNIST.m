clear
clc
setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 2: running NRPCA for MNIST dataset, for specific
% handwritten digits 4 and 9.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% MNIST variables declearation
N = 2000;   % number of samples in total
[orig_data, cmap] = load_MNIST49(N/2);
K = 6;     % number of neighbors (including data itself)
P = size(orig_data,2);  % original data dimension
num_run = 5; % maximum rounds
niter = 150; % maximum iterations per round 

%% Running NRPCA multiple rounds
C = run_NRPCA(orig_data, K, num_run, niter, 0);
L1 = C{1};
L2 = C{2};
L3 = C{3};
L4 = C{4};
L5 = C{5};
%% Image Denoising Results (Comparison between diffenrent rounds)
perm = randperm(N,54);
figure()
subplot(1,3,1), display_network(orig_data(perm,:)');title('Noisy images','fontsize',15);
subplot(1,3,2), display_network(L1(perm,:)');title('Denoised images','fontsize',15);
subplot(1,3,3), display_network(L2(perm,:)');title('Denoised images with neighbor updated','fontsize',15);

%% Dimension Reduction using estimated clean data matrix via LLE, Isomap, Laplacian Eigenmap
M=10;
% LLE
mapped_L1_lle = compute_mapping(L1,'LLE',2,M);
mapped_L2_lle = compute_mapping(L2,'LLE',2,M);
mapped_orig_lle = compute_mapping(orig_data,'LLE',2,M);

% Laplacian Eigenmap
mapped_L1_lap = compute_mapping(L1,'Laplacian',2,M);
mapped_L2_lap = compute_mapping(L2,'Laplacian',2,M);
mapped_orig_lap = compute_mapping(orig_data,'Laplacian',2,M);

% Isomap
mapped_L1_iso = compute_mapping(L1,'Isomap',2,M);
mapped_L2_iso = compute_mapping(L2,'Isomap',2,M);
mapped_orig_iso = compute_mapping(orig_data,'Isomap',2,M);

%% Visualization of embedding results
perm =randperm(N); % plotting in random order

% LLE
figure()
subplot(1,3,2), scatter(mapped_L1_lle(perm,1), mapped_L1_lle(perm,2),10, cmap(perm)), title('NRPCA denoised, LLE');xlabel('\fontsize{14}LLE1');ylabel('\fontsize{14}LLE2');
subplot(1,3,3), scatter(mapped_L2_lle(perm,1), mapped_L2_lle(perm,2),10, cmap(perm)), title('NRPCA denoised, with neighbor updated, LLE');xlabel('\fontsize{14}LLE1');ylabel('\fontsize{14}LLE2');
subplot(1,3,1), scatter(mapped_orig_lle(perm,1), mapped_orig_lle(perm,2),10, cmap(perm)), title('Noisy data, LLE');xlabel('\fontsize{14}LLE1');ylabel('\fontsize{14}LLE2');

% Laplacian Eigenmap
figure()
subplot(1,3,2), scatter(mapped_L1_lap(perm,1), mapped_L1_lap(perm,2), 10, cmap(perm)), title('NRPCA denoised, Laplacian');xlabel('\fontsize{14}Laplacian1');ylabel('\fontsize{14}Laplacian2');
subplot(1,3,3), scatter(mapped_L2_lap(perm,1), mapped_L2_lap(perm,2), 10, cmap(perm)), title('NRPCA denoised, with neighbor updated, Laplacian');xlabel('\fontsize{14}Laplacian1');ylabel('\fontsize{14}Laplacian2');
subplot(1,3,1), scatter(mapped_orig_lap(perm,1), mapped_orig_lap(perm,2), 10, cmap(perm)), title('Noisy data, Laplacian');xlabel('\fontsize{14}Laplacian1');ylabel('\fontsize{14}Laplacian2');

% Isomap
figure()
subplot(1,3,2), scatter(mapped_L1_iso(perm,1), mapped_L1_iso(perm,2), 10, cmap(perm)), title('NRPCA denoised, Isomap');xlabel('\fontsize{14}Isomap1');ylabel('\fontsize{14}Isomap2');
subplot(1,3,3), scatter(mapped_L2_iso(perm,1), mapped_L2_iso(perm,2), 10, cmap(perm)), title('NRPCA denoised, with neighbor updated, Isomap');xlabel('\fontsize{14}Isomap1');ylabel('\fontsize{14}Isomap2');
subplot(1,3,1), scatter(mapped_orig_iso(perm,1), mapped_orig_iso(perm,2), 10, cmap(perm)), title('Noisy data, Isomap');xlabel('\fontsize{14}Isomap1');ylabel('\fontsize{14}Isomap2');

%% Ploting Embedding results with digital images
mapped_data = mapped_orig_iso;
figure;scatter(mapped_data(:,1),mapped_data(:,2),10,cmap); hold on
cmap1 = [autumn(N/2+1); white(N/2-1); gray(N/2+1)];
colormap(cmap1);caxis([0,3])
perm = randperm(N);
Xmax = max(mapped_data(:,1)); Xmin = min(mapped_data(:,1));
dx = (Xmax-Xmin)/70; dy = dx; 
for i = 1:N  
    % rescale image, to get visualizable results
    xmin = mapped_data(perm(i),1)-dx; xmax = mapped_data(perm(i),1)+dx;
    ymin = mapped_data(perm(i),2)-dy; ymax = mapped_data(perm(i),2)+dy;
    img =  flip((reshape(orig_data(perm(i),:),28,28)));
    if perm(i) > N/2
        img = img+2*ones(28,28);
    end
    h = imagesc([xmin, xmax],[ymin, ymax],img);   
end
hold off
grid on

%% Image Denoising Results (comparison between different classes)
perm4 = randperm(N/2,45);
figure()
subplot(2,2,1), display_network(orig_data(perm4,:)');title('Original images for digit 4','fontsize',28);
subplot(2,2,2), display_network(L2(perm4,:)');title('Denoised images for digit 4','fontsize',28);

perm9 = N/2 + randperm(N/2,45); 
subplot(2,2,3), display_network(orig_data(perm9,:)');title('Original images for digit 9','fontsize',28);
subplot(2,2,4), display_network(L2(perm9,:)');title('Denoised images for digit 9','fontsize',28);
%% Image Denoising Results for T = 1 : 5
perm1 = perm(1:54);
figure()
subplot(2,3,1), display_network(orig_data(perm1,:)');title('Original images','fontsize',15);
subplot(2,3,2), display_network(L1(perm1,:)');title('Denoised images with T=1','fontsize',15);
subplot(2,3,3), display_network(L2(perm1,:)');title('Denoised images with T=2','fontsize',15);
subplot(2,3,4), display_network(L3(perm1,:)');title('Denoised images with T=3','fontsize',15);
subplot(2,3,5), display_network(L4(perm1,:)');title('Denoised images with T=4','fontsize',15);
subplot(2,3,6), display_network(L5(perm1,:)');title('Denoised images with T=5','fontsize',15);