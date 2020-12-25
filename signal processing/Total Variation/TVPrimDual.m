
function [u_out, output] = TVPrimDual(A, b, beta, alpha, u_init, iter, tol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [u_out, output] = TVPrimDual(A, b, beta, alpha, u_init, iter, tol)
%
% TVPrimDual.m implements an algorithm of Chan, Golub and Mulet to
% perform total variation regularization on an image deblurring problem.
%
% The algorithm uses a primal-dual Newton Method.  It uses
% conjugate gradients compute the search direction for the
% primal variables and includes a line search to determine
% the primal steplength.
% 
% Reference: Tony F. Chan, Gene H. Golub and Pep Mulet. "A Non-
% linear Primal-Dual Method for Total Variation-Based Image 
% Restoration,"  SIAM J. Sci. Comp., 20(6), 1999, pp.1964–1977.
%
% We want to solve: 
%       min_u   alpha*R(u)   +   .5*|| A*u-b ||_2^2
% where R(v) is the total variation 
%        R(v)  = sqrt ( | grad v |^2 + beta), 
% summed over all gridpoints, and alpha and beta are parameters.
% 
% Inputs:
%       A : defines the blurring.  It is either 
%             (a) a full or sparse matrix of dimension mn x mn
%             (b) a matrix object (defined in MyBR.m in RestoreTool)
%                 to perform matrix*vector and matrix'*vector operations
%       b : blurred image of size m x n.
%       beta: the flux parameter needed to deal with a singularity 
%             in the formulation of the problem. 
%       alpha: the regularization pameter.  
% 
% Optional Inputs: 
%       u_init: an initial guess for the solution  (Default: b)
%       iter:   max number of Newton iterations    (Default: 100)
%       tol:    tolerance on the Newton iterations (Default: 0.1)
%
% Outputs:
%      u_out : computed solution
%     output : structure with the following fields:
%      iterations - stopping iteration (options.Iter | GCV-determined)
%            Rnrm - relative residual norms
%            Xnrm - relative solution norms
%            flag - a flag that describes the output/stopping condition:
%                       0 - convergence attained.
%                       1 - performed max number of iterations
% In addition, there is a common variable called "pass" that
% gives global access to the solution, used for easy access outside
% the GIDE gui. In particular, pass.u_out = u_out.
%
% Organization of the software, with references to equation in the
% Chan-Golub-Mulet paper:
%
%  TVPrimDual.m Runs the Newton iteration for the primal-dual 
%               problem.
%  Cmult.m      Multiplies a vector by the C matrix, the matrix
%               in (3.5).
%  Dmult.m      Multiplies a vector by D (del) or D' (grad). 
%  formE.m      Computes a diagonal matrix with the norm of the
%               gradient of u at each gridpoint, and the inverse
%               of the matrix.
%  formF.m      Computes a block diagonal matrix F, defined
%               by (I - w grad-u' / | grad-u |) in (3.5).
%  DualStep.m   Evaluates (3.6) for delta_w.
%  fcn.m        Evaluates the function in (2.8) and its gradient in
%               (2.9) for the line search cvsrch and other purposes.
%  findsd.m     Solves (5.3) for largest feasible step.
%  cvsrch.m     Computes a (primal variable) step length via line search.
%  cstep.m      Used by cvsrch.
%
% Brianna Cash and Dianne O'Leary 06/2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global pass

% Set default parameters.

dTol = 1.e-5;  % default tolerance 
dIter = 100;   % default max number of Newton iterations.

if nargin < 4
    error('Not Enough Inputs')
end

if (nargin < 5)
    u_init = b; 
end
if (nargin < 6)
    iter = dIter; 
end
if (nargin < 7)
    tol = dTol;
end

if isempty(u_init)
    u_init = b;
end

if isempty(iter)
    iter = dIter;
end
if isempty(tol)
    tol = dTol;
end

% The following initialization is needed for RestoreTools 
% (taken from MyBR.m).
 
if isa(A, 'psfMatrix')
    bSize = size(b);
    A.imsize = bSize;
end
 
% Set up structures "pass" for common storage and "output" 
% for returned variables.

pass.alpha  = alpha;
pass.beta   = beta;
pass.A      = A;
pass.b      = b(:);
pass.bSize  = size(b);
pass.m      = size(pass.b,1);

outputparam = nargout>1;

if outputparam
    output.iterations = [];
    output.Enrm       = [];
    output.Rnrm       = [];
    output.Xnrm       = [];
    output.flag       = 1;
end

% Initialize the Newton iteration to compute primal 
% variables u(n) and dual variables w(n),  n = 0,1,....
% Store the current iterate in un and wn.

m = pass.m;
p0 = u_init(:) - b(:);
un = u_init(:);
p = p0;

[f, G] = fcn(m, un);

rhoG_0 = norm(G);
rhoG_old = rhoG_0;
rhoG_new = rhoG_old;
maxcgit = 100;  % max number of cg iterations
sigma = 10^-4;
k = 0;

disp(sprintf('Total Variation Newton iteration %d, gradient norm = %e', k, rhoG_new))

% Initialize D'u, E, Einverse, and Fbar for use in Cmult.
% (They are also used in computing the update for the dual variables pass.wn.)

Dtu = Dmult(pass.bSize(1), pass.bSize(2), un, 1);

[e, einv] = formE(Dtu, pass);

wn = einv .* Dtu;
Fbar = formF(wn, Dtu, einv, pass);

% Perform Newton iteration.

while rhoG_new >tol*rhoG_0 && k<iter

    k=k+1;    

    % Compute the approximate Newton direction for the primal 
    % variables using conjugate gradients to solve C d = -G
    
    % Set the convergence tolerance tol_D for cg.
    % See the local convergence Theorem 6.1.1, page 95, in 
    % C. T. Kelley, Iterative Methods for Linear and 
    % Nonlinear Equations, vol. 16 of Frontiers in Applied Mathematics, 
    % SIAM, 1995. 

    tol_D = min(.0001, .9*rhoG_new^2/rhoG_old^2);
    
    [p, cg_flag, cg_relres, cg_iter] = ...
               pcg(@(v)Cmult(v, einv, Fbar, pass), -G, tol_D, maxcgit);

    % Update the primal variables using the Newton direction or
    % (if the approximate Newton direction is not sufficiently
    % downhill) the steepest descent direction -G.

    % If the approximate Newton direction is not sufficiently downhill,
    % use the steepest descent direction. 

     if -G'*p < 0.001*norm(G)*norm(p)
       'Taking steepest descent step.'
       p = -G;
     end 
           
%   disp(sprintf('cos of angle between -grad and p = %f', ...
%                -G'*p/(norm(G)*norm(p) ) ))
        
    info = 0;
    u_t = un + p;
    [f_t, G_t] = fcn(m, u_t);
    if abs(f_t) <abs((1-sigma)*f)
       f = f_t;
       G = G_t;
       un = u_t;
       stp = 1;
    else
       % Determine a steplength for u using the line search cvsrch.

       [un, f, G, stp, info, nfev] = ...
        cvsrch(@fcn, m, un, f, G, p, 1, 10^-4, .1, .1, 0, 1, 1000);
       disp(sprintf('                Primal step length = %f, info = %d, cg_flag= %d ', ...
                      stp, info, cg_flag))
    end

    if info == 2
       disp('Line search unable to find an acceptable step length.')
       break;
    end
    
    % Update the dual variables.

    q = DualStep(wn, p, Fbar, einv, Dtu, pass);
    eta = findsd(m, wn, q);

    wn = wn + eta*q;

    % Update E, Einverse, and Fbar.

    Dtu = Dmult(pass.bSize(1), pass.bSize(2), un, 1);
    [e, einv] = formE(Dtu, pass);

    Fbar = formF(wn, Dtu, einv, pass);

    rhoG_old = rhoG_new;
    rhoG_new = norm(G);
    disp(sprintf('Total Variation Newton iteration %d, gradient norm = %e', k, rhoG_new))

end % Newton iteration


% Prepare to return.

u_out = reshape(un, pass.bSize);

if outputparam
    output.iterations= k;
    output.Rnrm = norm(b(:) - A*un);
    output.Xnrm = norm(un(:));
    output.flag = rhoG_new > tol*rhoG_0;
end
