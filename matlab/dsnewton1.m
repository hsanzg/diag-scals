function [P, x, y, residual, iter] = dsnewton1(A, r, c, maxiter, tol)
%DSNEWTON1 Approximate diagonal scaling with prescribed row and column sums.
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
x = ones(m, 1) / m; y = ones(n, 1) / n;

for iter=1:maxiter
    xprev = x; yprev = y;
    u = A * y; v = A.' * x;
    B = [diag(u), x .* A;
         y .* A', diag(v);
         ones(1, m), -ones(1, n)];
    z = [u .* x + r; v .* y + c; 0];
    w = B\z;
    if any(w <= 0)
        warning('dsnewton1: got negative w_%d iterate', iter);
        residual = Inf;
        break
    end
    x = w(1:m); y = w(m + 1:end);

    residual = norm([x ./ xprev; y ./ yprev] - 1, Inf);
    if residual <= tol, break; end
end
P = x .* A .* y';
end
