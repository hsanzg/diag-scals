# Diagonal scalings of nonnegative matrices with relaxed targets for the row and column sums

A diagonal scaling of a real matrix $A$ with nonnegative entries is a product of
the form $XAY$, where $X$ and $Y$ are real diagonal matrices with positive entries on
the main diagonal. This repository contains C and MATLAB programs for three algorithms
to approximately compute diagonal scalings of a given nonnegative matrix that have
prescribed row and column sums.
The iterative schemes work under different stopping criteria, with two of them
being ideally suited to relaxed tolerances for the target row and column sums.

## Methods

### The explicit Sinkhorn–Knopp-like method

The explicit variant of the Sinkhorn–Knopp-like algorithm, devised by J. Kruithof;
see the paper [_Properties of Kruithof's Projection Method_](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1979.tb02231.x)
by R. S. Krupp for details. The convergence properties of the method in the doubly stochastic
case were analyzed by R. Sinkhorn and P. Knopp in [_Concerning nonnegative matrices and doubly stochastic matrices_](https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-21/issue-2/Concerning-nonnegative-matrices-and-doubly-stochastic-matrices/pjm/1102992505.full).

### The implicit Sinkhorn–Knopp-like method
The implicit variant of the Sinkhorn–Knopp-like algorithm, proposed by
B. Kalantari and L. Khachiyan in [_On the complexity of nonnegative-matrix scaling_](https://core.ac.uk/download/pdf/82614641.pdf).
The main ideas of the scheme are summarized in the paper [_The Sinkhorn-Knopp algorithm: convergence and applications_](https://strathprints.strath.ac.uk/19685/1/skapp.pdf) by P. A. Knight.

### Newton's diagonal scaling method
A generalization of Newton's method proposed by P. A. Knight and D. Ruiz in
Section 2 of [_A fast algorithm for matrix balancing_](https://d-nb.info/991914708/34)
for balancing symmetric, nonnegative matrices to the general diagonal scaling problem.

## Stopping criteria
Let $X^{(k)}$ and $Y^{(k)}$ be respectively the approximations to the matrices
$X$ and $Y$ at the $k$th step of an iterative diagonal scaling method.
We can write $X^{(k + 1)} = \Gamma^{(k)} X^{(k)}$ and $Y^{(k + 1)} = \Delta^{(k)} Y^{(k)}$,
where $\Gamma^{(k)}$ and $\Delta^{(k)}$ are diagonal matrices with positive main diagonal.
Then an iterative process may stop whenever

1. the matrix $\Gamma^{(k)} \oplus \Delta^{(k)}$ is near the identity in the $l_\infty$ norm,
2. the spectral condition number of $\Gamma^{(k)} \oplus \Delta^{(k)}$ is close to 1, or
3. the maximum of the spectral condition numbers of $\Gamma^{(k)}$ and $\Delta^{(k)}$ is
   close to 1;
or after a maximum number of iterations have been run.

## License

[MIT](LICENSE) &copy; [Hugo Sanz González](https://hsgs.me)