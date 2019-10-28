function curvature = overall_curvature(X,normalize)
% X: nxp data matrix
% normalize: 0 actual curvature
%            1 normalized curvature
n = size(X,1);
if normalize ==1 
    X = X - mean(X,2);
    X = X / max(sqrt(diag(X*X')));
end
[A,D]=distance_matrix(X,20); %compute the adjacency and distance matrix
%% define r1 and r2
no_p = 50;
no_pairs = 100;
points = randsample(1:n,no_p);
[Dg,n_step] = dijks(A,points,1:n);
r1 = 4; r2=15;
D1 = D(points,:);
adm_pairs = find(n_step<r2 & n_step>r1&  D1>3e-1);
if length(adm_pairs(:))<no_pairs
    no_p = 50;
    points = randsample(1:n,no_p);
    [Dg,n_step] = dijks(A,points,1:n);
    r1 = 2; r2=15;
    adm_pairs = find(n_step<r2 & n_step>r1 & D1>3e-1);
end
if length(adm_pairs(:))<no_pairs
    warning('not enough points, estimation may be inaccurate');
    no_pairs = length(adm_pairs(:));
end   
sample_ind=randsample(adm_pairs,no_pairs); % randomly pick a subset of pairs
[row, col] = ind2sub(size(Dg),sample_ind);

%% estimate curvature
 options = optimset('Display','off','TolX',1e-16, 'TolPCG',1e-3);
 r_est = -ones(no_pairs,1);
 for i=1:no_pairs
     dg = Dg(row(i),col(i));
     du = D(points(row(i)),col(i));
      s = max(dg/du,1);
        if s < Inf  %prevent the situation that there is no path between these two points
            F = @(x)(sin(x)/(x) - 1/s);
            p = [1/120,-1/6,1-1/s];
            r = roots(p);
            r = sqrt(r(find(r>0)));
            theta1 = fsolve(F,sqrt((1-1/s+1e-20)*6),options);
            theta2 = fsolve(F,sqrt((1-1/s+1e-20)*6),options);
            if abs(F(theta1))<abs(F(theta2))
                theta = theta1;
            else
                theta = theta2;
            end
            if theta >= 0
              r_est(i) = dg/(2*theta);
            end
        end
 end
 curvature = sqrt(mean(max(realmin,r_est(r_est >= 0)).^(-2)));
end
         
                       
