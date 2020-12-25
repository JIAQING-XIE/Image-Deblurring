function Cv = Cmult(v, einv, Fbar, pass)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function Cv = Cmult(v, einv, Fbar, pass)
% This function is used by TVPrimDual.m in solving the primal-dual 
% total variation problem.
%
% Cmult computes  Cv = .5 * (C + C')*v, 
% where
% C = A' * A + pass.alpha * D * E^{-1} * F_bar * D'  
% is the matrix in (3.5) in the Chan-Golub-Mulet paper.
% Brianna Cash, Dianne O'Leary 06/2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Dtv = Dmult(pass.bSize(1), pass.bSize(2), v, 1);  % "1" for "transpose"

FEDtv = einv .* Dtv;

for i = 2:2:2*pass.m
    FDtv(i-1:i,1)  = Fbar(:,i-1:i)  *   Dtv(i-1:i);
    FEDtv(i-1:i,1) = Fbar(:,i-1:i)' * FEDtv(i-1:i);
end
    
% Note the use of (einv.*FDtv)+FEDtv to symmetrize E^{-1} F_bar.

Cv =  Dmult(pass.bSize(1), pass.bSize(2),(einv.*FDtv)+FEDtv, 0);
                                                  % "0" for no transpose

% No need to symmetrize the A'A part.

AAv= pass.A  * v;
AAv= pass.A' * AAv;

% Add the two pieces together.

Cv = AAv + .5 * pass.alpha * Cv; 
