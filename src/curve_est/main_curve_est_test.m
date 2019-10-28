%% Test the curvature estimation algorithms
%% Example: sphere
clear all
S = randn(2000,3);
S = normc(S')';
fprintf('3 dimmensional unit sphere: true curvature 1, estimated %d \n', overall_curvature(S,0))
S = randn(2000,10);
S = normc(S')';
fprintf('10 dimmensional unit sphere: true curvature 1, estimated %d \n', overall_curvature(S,0))
S = randn(1000,5);
S = 3*normc(S')';
fprintf('5 dimmensional sphere with radius 3: true curvature 1/3, estimated %d \n', overall_curvature(S,0))
%% Example: plane
t = .001:.001:1;
data = [t',2*t'];
fprintf('plane: true curvature 0, estimated %d \n', overall_curvature(data,0))
%% Example: periodic curve 
t =  .001:.001:1;
data = [t;cos(10*(t+1).^2)];
r = curvature(data');
figure(1); subplot(2,1,1);plot(t,data(2,:));title('data');  subplot(2,1,2);plot(t,r);title('estimated curvature');
%% Example: Swiss roll
load('../../data/swissroll_clean.mat')
r = curvature(clean_data(:,1:2));
figure;plot(r); title('estimated curvature of the swiss roll');

load('../../data/digit_9.mat')
overall_curvature(digit_9',1)
