function curvature = curvature(X)
% estimate pointwise curvature
N = size(X,1);
%d = intrinsic_dim(X,'GMST');
%[A,D]=distance_matrix(X,round(min(3*2^(d),N*.1))); %compute the adjacency and distance matrix
[A,D]=distance_matrix(X,20);
S=[];
curvature = zeros(N,1);
while size(S,1) < N
    % estimating geodesic distance using dijkstra's algorithm
    center = randsample(setdiff(1:N, S),1);
    % find neighbors located in a larger radius
    index = find(D(center,:) <= (prctile(D(center,:),10)));
    pair = min(size(index,2),500);
    index = index(:,1:pair);
    R = curv_omega(D(index,index),A(index,index));
    
    index_new = setdiff(index,S);
    curvature(index_new) = R;
    S = union(S,index_new);
end
end
function curvature = curv_omega(D,A) 
n = size(D,1);
no_p = min(40,n);
no_pairs = 100;
points = randsample(1:n,no_p);
[Dg,n_step] = dijks(A,points,1:n);
r1 = 4; 
D1 = D(points,:);
adm_pairs = find( n_step>r1 & D1>1e-2);

if length(adm_pairs(:)) < no_pairs 
    r1 = 2; 
    D1 = D(points,:);
    adm_pairs = find( n_step>r1 & D1>1e-2);
end
if length(adm_pairs(:)) < no_pairs 
    no_pairs/4;
    r1 = 2; 
    D1 = D(points,:);
    adm_pairs = find( n_step>r1 & D1>1e-2);
end
if length(adm_pairs(:)) < no_pairs 
    warning('not enough points, curvature may be inaccurate')
    r1 = 0; 
    D1 = D(points,:);
    adm_pairs = find( n_step>r1 & D1>1e-2);
    no_pairs= length(adm_pairs(:));
    [row, col] = ind2sub(size(Dg),adm_pairs);
else
sample_ind=randsample(adm_pairs,no_pairs); % randomly pick a subset of pairs
[row, col] = ind2sub(size(Dg),sample_ind);
end

%% estimate curvature
 options = optimset('Display','off','TolX',1e-16, 'TolPCG',1e-3);
 r_est = -ones(no_pairs,1);
 for i=1:no_pairs
     dg = Dg(row(i),col(i));
     du = D(points(row(i)),col(i));
      s = max(dg/du,1);
        if s < Inf  %prevent the situation that there is no path between these two points
            % F = @(x)(x/sin(x) - s);
          F = @(x)(sin(x)/(x) - 1/s);
            theta = fsolve(F,abs((sqrt((1-1/s+1e-30)*6))),options);
            if theta >= 0
              r_est(i) = dg/(2*theta);
            end
        end
 end
 curvature = sqrt(mean(max(realmin,r_est(r_est > 0)).^(-2)));

end
         
                       
