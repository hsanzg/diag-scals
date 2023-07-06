function [P, x, y, residual, iter] = dsimplicit1(A, r, c, maxiter, tol)
%DSIMPLICIT1 Approximate diagonal scaling with prescribed row and column sums.
% Returns a diagonal scaling P=diag(x)*A*diag(y) of the nonnegative
% matrix A with row and column sums approximately equal to r and c,
% respectively.
%
% Inputs:
% A: nonnegative mxn matrix.
% r: positive mx1 column vector.
% c: positive nx1 column vector such that sum(r)=sum(c).
% maxiter: maximum number of iterations.
% tol: tolerance parameter; see below.
%
% Outputs:
% P: nonnegative mxn matrix.
% x: positive mx1 column vector.
% y: positive nx1 column vector.
% residual: error residual in the approximation; see below.
% iter: number of iterations performed.
%
% Stopping criterion:
% The iterative process stops when the l_\infty norm of the vector
% [x_k ./ x_{k-1}; y_k ./ y_{k-1}] - ones(m+n,1) is less than or
% equal to tol, or after maxiter iterations have been run.

[m, n] = size(A);
x = ones(m, 1); y = ones(n, 1);

for iter=1:maxiter
    yprev = y;
    y = c ./ (A.' * x);
    xprev = x;
    x = r ./ (A * y);

    residual = norm([x ./ xprev; y ./ yprev] - 1, Inf);
    if residual <= tol, break; end
end
P = x .* A .* y.';
end
