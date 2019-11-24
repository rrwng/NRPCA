classdef solver_patchRPCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% solver_patchRPCA is a solver for the optimization problem
% minimize sum_{i=1}^n (||X^i - L^i - S^i||_F^2 + alpha ||L^i||_*) 
% + lambda ||S||_1
%
% parameters
%   N: number of samples
%   K: number of neighbors (including itself)
%   P: original dimension of data
%   X^i : P * K, local patch of noisy data
%   S : P * N, estimates sparse noise
%   S^i: P * K, local patch of sparse noise
%
% (C) Ningyu Sha, Michigan State University
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

properties
    para;
    gradf = @(Xi, Si, entry)     0;
    proxg = @(Si, t)  Si;
end

methods
    function [S, fcnvalue] = loss_minimizer(this)
            
        % check initial values
        if ~isfield(this.para, 'Si')
            error('Error. Put the initial sparse matrix S in this.para.Si')               
        end 
        Si   = this.para.Si;
           
        if isfield(this.para, 'iter')
            iter  = this.para.iter;
        else
           iter = 1000;
        end     
            
        if ~isfield(this.para, 't')
            error('Error. Put the stepsize t')               
        end      
        t   = this.para.t;
            
        Xi   = this.para.Xi;
        entry = this.para.entry;

        % using FISTA to solve the problem
        fcnvalue = zeros(iter, 1);
        theta   = 1; 
        S_old   = Si;
        for i = 1 : iter
            S_new = this.proxg(Si - t * this.gradf(Xi, Si, entry), t);
            theta_new = 0.5 * (1 + sqrt(1 + 4 * theta^2));
            Si    = S_new + (theta - 1)/theta_new * (S_new - S_old);
            S_old   = S_new; 
            theta   = theta_new;
        end
        S = S_new;
    end
end
    
end