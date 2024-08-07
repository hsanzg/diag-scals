# diag-scals

[![Build status](https://img.shields.io/github/actions/workflow/status/hsanzg/diag-scals/docs-gh-pages.yml)](https://github.com/hsanzg/diag-scals/actions/)
[![C documentation](https://img.shields.io/badge/docs-passing-brightgreen)](https://hsanzg.github.io/diag-scals/)

A diagonal scaling of a real matrix $A$ with nonnegative entries is a product of
the form $XAY$, where $X$ and $Y$ are real diagonal matrices with positive entries on
the main diagonal. This repository contains C and MATLAB subroutines for three algorithms
that approximately compute diagonal scalings of a given nonnegative matrix with
prescribed row and column sums.
The iterative schemes work under different stopping criteria, with two of them
being ideally suited to relaxed tolerances for the target row and column sums.

This work is part of my [undergraduate thesis](https://hgsg.me/bachelor/) in Applied Mathematics and Computing at the [University Carlos III of Madrid](https://www.uc3m.es/Home).

## Methods

### The explicit Sinkhorn–Knopp-like method

The explicit variant of the Sinkhorn–Knopp-like algorithm, devised by J. Kruithof;
see the paper [_Properties of Kruithof's Projection Method_](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1979.tb02231.x)
by R. S. Krupp for details. The convergence properties of the method in the doubly stochastic
case were analyzed by R. Sinkhorn and P. Knopp in [_Concerning nonnegative matrices and doubly stochastic matrices_](https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-21/issue-2/Concerning-nonnegative-matrices-and-doubly-stochastic-matrices/pjm/1102992505.full).

### The implicit Sinkhorn–Knopp-like method

The implicit variant of the Sinkhorn–Knopp-like algorithm, proposed by
B. Kalantari and L. Khachiyan in the paper [_On the complexity of nonnegative-matrix scaling_](https://core.ac.uk/download/pdf/82614641.pdf).
The main ideas of the scheme are summarized in [_The Sinkhorn-Knopp algorithm: convergence and applications_](https://strathprints.strath.ac.uk/19685/1/skapp.pdf) by P. A. Knight.

### Newton's diagonal scaling method

A novel extension of P. A. Knight and D. Ruiz's symmetric matrix balancing method
from Section 2 of [_A fast algorithm for matrix balancing_](https://d-nb.info/991914708/34)
to the general diagonal scaling problem. See Section 4.2 of [my thesis](https://hgsg.me/bachelor/)
for further information.

## Stopping criteria

Let $X^{(k)}$ and $Y^{(k)}$ be respectively the approximations to the matrices
$X$ and $Y$ at the $k$th step of an iterative diagonal scaling method.
We can write $X^{(k + 1)} = \Gamma^{(k)} X^{(k)}$ and $Y^{(k + 1)} = \Delta^{(k)} Y^{(k)}$,
where $\Gamma^{(k)}$ and $\Delta^{(k)}$ are diagonal matrices with positive main diagonals.
This library provides implementations of the iterative processes listed above that stop whenever

1. the matrix $\Gamma^{(k)} \oplus \Delta^{(k)}$ is near the identity in the $l_\infty$ norm,
2. the spectral condition number of $\Gamma^{(k)} \oplus \Delta^{(k)}$ is close to 1, or
3. the maximum of the spectral condition numbers of $\Gamma^{(k)}$ and $\Delta^{(k)}$ is
   close to 1;

or after a maximum number of iterations have been run.

## Usage

You will need the following dependencies to build and use the C library:
- A C11 compiler,
- cmake ≥ 3,
- Intel oneMKL ≥ 2021.1 (provides BLAS and LAPACK implementations),
- An x86 processor with support for AVX2 instructions.

```bash
git clone git@github.com:hsanzg/diag-scals.git
cd diag-scals/c
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKING=OFF -Bbuild
cmake --build build --parallel
```

The MATLAB scripts have been tested on version R2022b, and they accept matrices stored in either dense or sparse form as input.

## Examples

The following C test program uses the explicit Sinkhorn–Knopp-like algorithm under the first stopping criterion to approximately balance a $100 \times 100$ matrix of random floating point numbers in $[1, 10)$:
```c
#include <stdlib.h>
#include <stdio.h>
#include <diag_scals/diag_scals.h>

int main(void) {
    const size_t n = 100;
    ds_problem pr;
    ds_problem_init(&pr, n, n, /* max_iters */ 10, /* tol */ 1e-4);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            // Store matrices in column-major order.
            pr.a[n * j + i] = 1. + 9. * rand() / RAND_MAX;
    for (size_t i = 0; i < n; ++i) pr.r[i] = 1.;
    for (size_t j = 0; j < n; ++j) pr.c[j] = 1.;
    
    ds_sol sol = ds_expl_crit1(pr); // clobbers pr.a
    if (ds_sol_is_err(&sol)) return 1;
    
    // Print the approximate solution; here P = XAY.
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n; ++i)
            printf("%lf ", sol.p[n * j + i]);
        putchar('\n');
    }
    
    ds_sol_free(&sol);
    ds_problem_free(&pr);
    
    // Clean up the working area resources.
    ds_work_area_free();
    return 0;
}
```

The [C library documentation](https://hsanzg.github.io/diag-scals/) contains
current information about all available subroutines and data structures.

The corresponding MATLAB program is quite simple:
```matlab
n = 100;
A = 1 + 9 * rand(n);
r = ones(n, 1); c = r;

[P, x, y, residual, iters] = dsexplicit1(A, r, c, 10, 1e-4);
P
```

## License

[MIT](LICENSE) &copy; [Hugo Sanz González](https://hgsg.me)
