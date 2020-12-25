function [e, einv] = formE(Dtu, pass)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [e, einv] = formE(Dtu, pass)
% This function is used by TVPrimDual.m in solving the primal-dual 
% total variation problem.
%
% The matrix E is diagonal and contains (repeated) entries equal to
%       sqrt( | grad u |^2 + pass.beta ), 
% where two consecutive entries of v contain grad u at one mesh point.
% See the Chan-Golub-Mulet paper.
% Input:
%    Dtu  -  grad u: a vector of length 2*pass.m x 1
%    pass -  a structure containing a scalar pass.beta 
%               and a dimension pass.m
% Output:
%   e    - vector of diagonal entries in E  (dimension 2m)
%   einv - vector of inverses of the elements in e.
% Brianna Cash and Dianne O'Leary 06/2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

beta = pass.beta;
m    = pass.m;

e = zeros(2*m, 1);

for i = 2:2:2*m
    e(i-1) = sqrt(Dtu(i-1)^2 + Dtu(i)^2 + beta);
    e(i) = e(i-1);
end

einv = 1./e;
