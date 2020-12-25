function [Result] = Dmult(m, n, u, Transpose)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [Result] = Dmult(m, n, u, Transpose)
% This function is used by TVPrimResultal.m in solving the primal-dual 
% total variation problem.
%
% It constructs the discrete 'grad' operator D', the matrix which, applied
% to an array of function values over a grid, computes the gradient at
% grid point.
%
% It can also apply the adjoint D of this operator, the 'del' operator.
% 
% It uses forward differencing in horizontal and vertical directions.
% See the Chan-Golub-Mulet paper.
%
% For D', a 2mn x mn matrix,
% the two rows corresponding to the ith mesh point are
% D_{i}*u = [u_{i+n} - u_{i};   (horizontal difference)
%            u_{i+1} - u_{i}]   (vertical difference)
% At the boundary we use Neumann boundary conditions.
%
% Input: 
%       m - number of rows in the grid 
%       n - number of columns in the grid 
%       x - a column vector of length m*n   (if Transpose = 1) 
%                          or       2*m*n   (if Transpose = 0)
% Output: 
%       If Transpose = 0: 
%          Result = D'*u   (column vector of length 2mn)
%       If Transpose = 1: 
%          Result = D*u    (column vector of length mn)
%
% Brianna Cash and Dianne O'Leary 06/2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if Transpose

   U = reshape(u,m,n);   % Reshape the vector as a grid.

   dh = zeros(m,n);      % differences in horizontal direction
   dh(:,1:n-1) = U(:,2:n) - U(:,1:n-1);  % Odd  entries in Dtx
   dh(:,n)     =  -dh(:,n-1);  % Neumann boundary condition

   dv = zeros(m,n);      % differences in vertical   direction
   dv(1:m-1,:) = U(2:m,:) - U(1:m-1,:);  % Even entries in Dtx 
   dv(m,:)     =  -dv(m-1,:);  % Neumann boundary condition

   Result(1:2:2*m*n,1) = dh(:);  % Odd  entries in the result
   Result(2:2:2*m*n,1) = dv(:);  % Even entries in the result

else

   dh = u(1:2:end);     % Odd  entries in u
   dh = reshape(dh,m,n);

   dv = u(2:2:end);     % Even entries in u
   dv = reshape(dv,m,n);

   Result = zeros(m,n);

   Result = - (dh + dv);
   Result(2:m,:) = Result(2:m,:) + dv(1:m-1,:);
   Result(:,2:n) = Result(:,2:n) + dh(:,1:n-1);

   Result(m-1,:) = Result(m-1,:) + dv(m,:);
   Result(:,n-1) = Result(:,n-1) + dh(:,n);

   Result = Result(:);  % Reshape the grid as a vector.

end
