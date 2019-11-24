classdef solver_NRPCA
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% minimize sum_{i=1}^n (||X^i - L^i - S^i||_F^2) + lambda ||S||_1
%
% Parameters
%   X^i : P * K, P is original dimension, K is number of neighbors, patchwise
%   noisy data matrix
%   L^i: P * K, local data matrix
%   S^i: P * K, local sparse noise matrix
%   S : P * N, N is number of data set, sparse noise matrix
%
% (C) Ningyu Sha, Michigan State University
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

properties
    para;
    funf = @(S)      0;
    fung = @(S)      0;
    gradf = @(S)     0;
    proxg = @(S, t)  S;
end
    
methods
    function [L, fcnvalue] = loss_minimizer(this)
        % check initial values
        if ~isfield(this.para, 'S')
            error('Error. Put the initial sparse matrix S in this.para.S')
        end
        S   = this.para.S;
            
        if isfield(this.para, 'iter')
            iter  = this.para.iter;
        else
            iter = 1000;
        end
            
        if ~isfield(this.para, 't')
            error('Error. Put the stepsize t')
        end
        t   = this.para.t;
            
        if ~isfield(this.para, 'X')
            error('Error. Put the initial noisy matrix X in this.para.X')
        end
        X   = this.para.X;
            
        % using FISTA to solve the problem
        fcnvalue = zeros(iter, 1);
        theta   = 1;
        S_old   = S;
        for i = 1 : iter
            fprintf('Iteration: %d\n', i)
            S_new = this.proxg(S - t * this.gradf(S), t);
            theta_new = 0.5 * (1 + sqrt(1 + 4 * theta^2));
            S    = S_new + (theta - 1)/theta_new * (S_new - S_old);
            S_old   = S_new;
            theta   = theta_new;
            fcnvalue(i) = this.funf(S_new) + this.fung(S_new);
            fprintf('Objective Function Value: %e\n', fcnvalue(i));
            if i >= 2
                if abs(fcnvalue(i)-fcnvalue(i-1)) < 1e-5
                    break;
                end
            end
        end
        
        % remove sparse noise
        L = X - S_new;
    end
end
    
end