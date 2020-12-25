function f = paramtest(alpha, A, b, beta, nu, delta, maxit, tol)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% f = paramtest(alpha, A, b, beta, nu, delta, maxit, tol)
%
% This function is called by Matlab's fzero so that we can
% find a solution to the equation
% f(alpha) = norm(A*x(:) - b(:), 2)-nu*delta = 0,
% where x is the reconstructed image using alpha as a
% total variation regularization parameter.
%
% This is used in GIDE to find an initial value of alpha,
% based on the discrepancy principle, which says that the 
% norm of the residual should be about nu*delta.
%
% Brianna Cash, 2012
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the TV solution x , using alpha as the regularization parameter.

x = TVPrimDual(A, b, beta, alpha, [], maxit, tol);

% Compare the norm of the residual Ax-b to nu*delta.

f = norm(A*x(:) - b(:), 2)-nu*delta;

