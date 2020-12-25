function eta = findsd(m, y, dy)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function eta = findsd(m, y, dy)
% This function is used by TVPrimDual.m in solving the primal-dual 
% total variation problem.
% 
% This function finds the largest eta so that 
%             ||w_i + eta dw_i || <= 1 for all i,    (5.3)
% where w_i is the current value of the dual variable
% dw_i is the step direction. Both are 2-vectors extracted from y and dy.
% The equation number references the Chan-Golub-Mulet paper.
%
% We find eta by solving the quadratic equation
%        a_i eta^2 + b_i eta + c_i = 1  
% for each i and then taking the minimum.
% We set the final value of eta to .95 of the min of this value and 1.
%
% The coefficients a, b, and cm1 = c - 1 are defined in the code below.
% 
% by Brianna Cash and Dianne O'Leary 11/2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

s = zeros(m, 1);

% Compute the candidate values for eta and store them in s.

for j = 2:2:2*m

      a = dy(j-1:j)'*dy(j-1:j);
      b = 2 * dy(j-1:j)'*y(j-1:j);
      cm1 = y(j-1:j)'*y(j-1:j) - 1;
      s(j/2) = (-b + sqrt(b^2 - 4*a*cm1))/(2*a+eps);
      if (s(j/2) < 0)  % then this coordinate provides no restriction.
          s(j/2) = Inf;
      end

end

% Choose 0.95 times the largest step that keeps the norms bounded by 1.
% Note that negative values of s do not constrain the step.

eta = 0.95 *min(min(s(:)), 1);

