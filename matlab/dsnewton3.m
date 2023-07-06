function [P, x, y, residual, iter] = dsnewton3(A, r, c, maxiter, tol)
%DSNEWTON3 Approximate diagonal scaling with prescribed row and column sums.
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
% ignorenegiters: keep iterating even if some iterate of the diagonal
%                 matrices diag(x) or diag(y) contains a nonpositive entry
%                 on the main diagonal.
%
% Outputs:
% P: nonnegative mxn matrix.
% x: positive mx1 column vector.
% y: positive nx1 column vector.
% residual: error residual in the approximation; see below.
% iter: number of iterations performed.
%
% Stopping criterion:
% Let u_k = x_k ./ x_{k-1} and v_k = y_k ./ y_{k-1}. The iterative
% process stops when max{max u_k / min u_k, max v_k / min v_k} is
% less than or equal to (1+tol)/(1-tol), or after maxiter iterations have
% been run. To use the same criterion as in (F. M. Dopico et al., 2022),
% set tol = tol'/(4 - tol').

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
        warning('dsnewton3: got negative w_%d iterate', iter);
        residual = Inf;
        break
    end
    x = w(1:m); y = w(m + 1:end);

    [xminup, xmaxup] = bounds(x ./ xprev);
    [yminup, ymaxup] = bounds(y ./ yprev);
    residual = max(xmaxup / xminup, ymaxup / yminup);
    if residual <= (1 + tol)/(1 - tol), break; end
end
P = x .* A .* y.';
end
