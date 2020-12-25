function [f, g] = fcn(~, u)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [f, g] = fcn(~, u)
% This function is used by TVPrimDual.m in solving the primal-dual 
% total variation problem.
%
% It computes the function
%     f(u) =  alpha * sum(e) + .5 norm(A*u-b)^2,     (2.8)
% and its gradient
%     g(u) = - alpha*D* E^{-1}(u) D'u + A'(A*u-b).   (2.9)
% The equation numbers refer to the Chan-Golub-Mulet paper.
% It is designed to interface with cvsrch but is used elsewhere, too.
% Brianna Cash 2012
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global pass
resid = pass.A * u - pass.b;  % = Ku - z in Chan-Golub-Mulet notation

g = Dmult(pass.bSize(1), pass.bSize(2), u, 1);

[e, einv] = formE(g, pass);

g = einv.*g;

g = pass.alpha*Dmult(pass.bSize(1), pass.bSize(2), g, 0) + pass.A'*resid;

f = pass.alpha * sum(e) + 0.5 * norm(resid)^2;
    
