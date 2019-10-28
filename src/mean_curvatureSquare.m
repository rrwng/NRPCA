function  mean_squareCurvature = mean_curvatureSquare(pair, id, orig_data, len)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function mean_curvatureSquare estimates the mean square curvature of
% points on the manifold, and is used to compute parameter lambda.
%
% mean_squareCurvature = mean_curvatureSquare(pair, id, orig_data, len)
% Input:
%   pair: number of data sampled out
%   id: pairs of points on manifold where geodesic distance are calculated
%   orig_data: input data matrix with dimension N * P
%   len: geodesic distances estimated using dijkstra's algorithm
% Output:
%   mean_squareCurvature: estimated mean square curvature
%
% Author: He Lyu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

radius_est = zeros(pair,1);
options = optimset('Display','off');

for i = 1:pair
    % Euclidean distance 
    d = norm(orig_data(id(2*i-1),:) - orig_data(id(2*i),:),2);
    % Geodesic distance
    l = len(i);
    s = l/d;
    if s < 10000  %prevent the situation that there is no path between these two points
        F = @(x)(x/sin(x) - s);
        solvetheta = fsolve(F,rand(),options);
        if solvetheta < 0
            solvetheta = - solvetheta;
        end
        radius_est(i) = l/(2*solvetheta);
    end
end

mean_squareCurvature = mean(radius_est(radius_est ~= 0).^(-2));

end
