function [P, x, y, residual, iter] = dsnewton2(A, r, c, maxiter, tol)
%DSNEWTON2 Approximate diagonal scaling with prescribed row and column sums.
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
% Let v_k = [x_k ./ x_{k-1}; y_k ./ y_{k-1}]. The iterative process
% stops when the quotient of the largest entry of v_k and the smallest
% is less than or equal to (1+tol)/(1-tol), or after maxiter iterations
% have been run.

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
        warning('dsnewton2: got negative w_%d iterate', iter);
        residual = Inf;
        break
    end
    x = w(1:m); y = w(m + 1:end);

    [minup, maxup] = bounds([x ./ xprev; y ./ yprev]);
    residual = maxup / minup;
    if residual <= (1 + tol)/(1 - tol), break; end
end
P = x .* A .* y';
end
