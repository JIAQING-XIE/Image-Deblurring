function delta_w = DualStep(wn, p, Fbar, einv, Dtu, pass)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function delta_w = DualStep(pass)
% This function is used by TVPrimDual.m in solving the primal-dual 
% total variation problem.
% 
% It computes the dual step direction delta_w from equation (3.6)
% in the Chan-Golub-Mulet paper using
%      Dtu (the current D' * u),
%      p  (delta_u)
%      einv (the Einverse matrix)
%      Fbar (the F matrix)
% and  wn (the current w).
%
% Brianna Cash and Dianne O'Leary  06/2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


FDtp = Dmult(pass.bSize(1), pass.bSize(2), p, 1);

for i = 2:2:2*pass.m
    FDtp(i-1:i) = Fbar(:,i-1:i)*FDtp(i-1:i);
end

delta_w = einv.*(FDtp + Dtu) - wn;
   
