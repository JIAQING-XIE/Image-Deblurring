function [Fbar] = formF(wn, Dtu, einv, pass)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [Fbar] = formF(pass )
% This function is used by TVPrimDual.m in solving the primal-dual 
% total variation problem.
% The 2m x 2m matrix F is block diagonal, and we store its 2x2 blocks
% in the columns of Fbar, where, for i = 1, ..., m,
% 
%     Fbar(:,2*i-1:2*i) = I_2 - w_i gradu_i'/\nu_i 
%
% The values of gradu are stored in pass.Dtu.
%
% See the use of this matrix in equation (3.5) of the 
% Chan-Golub-Mulet paper.
% 
% Brianna Cash and Dianne O'Leary 06/2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m = pass.m;    

Fbar = zeros(2,2*m);

for i = 2:2:2*m 
   Fbar(:,i-1:i) = eye(2) - wn(i-1:i)*Dtu(i-1:i)'*einv(i); 
end

