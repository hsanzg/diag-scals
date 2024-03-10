/**
 * @file diag_scals.h
 *
 * This header defines the diagonal scaling structures and methods.
 */
#ifndef DIAGONALSCALINGS_DIAG_SCALS_H
#define DIAGONALSCALINGS_DIAG_SCALS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h> // size_t
#include <stdint.h> // uint32_t
#include <stdbool.h>

// The diagonal scaling problem.

/// The problem of finding a diagonal scaling of the mxn nonnegative matrix A
/// with row sums approximately equal to r and column sums approximately equal
/// to c within a tolerance `tol`, and a threshold of `max_iters` iterations.
///
/// A diagonal scaling method may clobber the matrix A, but the target row and
/// column sums vectors are left untouched.
typedef struct {
    double *__restrict a;
    double *__restrict r, *__restrict c;
    size_t m, n;
    double tol;
    uint32_t max_iters;
} ds_problem;

/// Creates an instance of the diagonal scaling problem.
///
/// The caller must check that all pointer members are nonnull after calling
/// this function. The diagonal scaling solvers do **not** check this precondition.
void ds_problem_init(ds_problem *pr, size_t m, size_t n,
                     uint32_t max_iters, double tol);

/// Deallocates the resources associated with the given problem.
/// IMPORTANT: this frees the matrix A, which might be shared with other
///            `ds_problem` or `ds_sol` instances. Take special care not
///            to reference or free `pr->a` after calling this function.
void ds_problem_free(ds_problem *pr);

/// An approximate solution to a diagonal scaling problem.
typedef struct {
    double *__restrict p;
    double *__restrict x, *__restrict y;
    double res; // residual
    uint32_t iters;
} ds_sol;

/// Creates a diagonal scaling solution object, leaving the pointer to
/// the P=X*A*Y matrix, the number of iterations, and the error set to zero.
///
/// The caller must check that all pointer members are nonnull after calling
/// this function. The diagonal scaling solvers do **not** check this precondition.
ds_sol ds_sol_init(const ds_problem *pr);

/// Deallocates the resources associated with the given solution.
/// IMPORTANT: the caller is responsible for freeing the matrix P, which might
///            be shared with other `ds_problem` or `ds_sol` instances.
void ds_sol_free(ds_sol *sol);

/// Indicates if an error occurred while running a diagonal scaling method,
/// and so the solution data is invalid and should not be referenced.
bool ds_sol_is_err(const ds_sol *sol);

/// Returns an address to an auxiliary array that can be used to store up to
/// `cap` double-precision floating point numbers. The contents in the work area
/// prior to this call may be discarded. The returned address is a multiple of
/// `DS_VEC_BYTE_WIDTH`.
///
/// This function may return `NULL` on allocation failure.
double *ds_work_area_get(size_t cap);

/// Cleans up all resources associated with the global work area.
/// The work area may be used even after it is freed.
void ds_work_area_free(void);


// Diagonal scaling methods.

// Implementations in explicit.c.
/// Computes a diagonal scaling `P=diag(x)*A*diag(y)` of the nonnegative matrix A
/// with row and column sums approximately equal to r and c.
///
/// # Implementation
/// The explicit variant of the Sinkhorn--Knopp-like algorithm, devised by
/// J. Kruithof; see the paper [_Properties of Kruithof's Projection Method_](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1979.tb02231.x)
/// by R. S. Krupp for details. The convergence properties in the doubly stochastic
/// case were analyzed by R. Sinkhorn and P. Knopp in [_Concerning nonnegative matrices and doubly stochastic matrices_](https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-21/issue-2/Concerning-nonnegative-matrices-and-doubly-stochastic-matrices/pjm/1102992505.full).
///
/// # Stopping criterion
/// The iterative process stops when the `l_\infty` norm of the vector
/// `[x_k ./ x_{k-1}; y_k ./ y_{k-1}] - ones(m+n,1)` is less than or equal to
/// `tol`, or after `max_iters` iterations have been run.
ds_sol ds_expl_crit1(ds_problem);

/// Computes a diagonal scaling `P=diag(x)*A*diag(y)` of the nonnegative matrix A
/// with row and column sums approximately equal to r and c.
///
/// # Implementation
/// The explicit variant of the Sinkhorn--Knopp-like algorithm, devised by
/// J. Kruithof; see the paper [_Properties of Kruithof's Projection Method_](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1979.tb02231.x)
/// by R. S. Krupp for details. The convergence properties in the doubly stochastic
/// case were analyzed by R. Sinkhorn and P. Knopp in [_Concerning nonnegative matrices and doubly stochastic matrices_](https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-21/issue-2/Concerning-nonnegative-matrices-and-doubly-stochastic-matrices/pjm/1102992505.full).
///
/// # Stopping criterion
/// Let `v_k = [x_k ./ x_{k-1}; y_k ./ y_{k-1}]`. The iterative process stops when
/// the quotient of the largest entry of v_k and the smallest is less than or
/// equal to `(1+tol)/(1-tol)`, or after `max_iters` iterations have been run.
ds_sol ds_expl_crit2(ds_problem);

/// Computes a diagonal scaling `P=diag(x)*A*diag(y)` of the nonnegative matrix A
/// with row and column sums approximately equal to r and c.
///
/// # Implementation
/// The explicit variant of the Sinkhorn--Knopp-like algorithm, devised by
/// J. Kruithof; see the paper [_Properties of Kruithof's Projection Method_](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1979.tb02231.x)
/// by R. S. Krupp for details. The convergence properties in the doubly stochastic
/// case were analyzed by R. Sinkhorn and P. Knopp in [_Concerning nonnegative matrices and doubly stochastic matrices_](https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-21/issue-2/Concerning-nonnegative-matrices-and-doubly-stochastic-matrices/pjm/1102992505.full).
///
/// # Stopping criterion
/// Let `u_k = x_k ./ x_{k-1}` and `v_k = y_k ./ y_{k-1}`. The iterative
/// process stops when `max{max u_k / min u_k, max v_k / min v_k}` is less than
/// or equal to `(1+tol)/(1-tol)`, or after `max_iters` iterations have been run.
/// To use the same criterion as in (F. M. Dopico et al., 2022), set
/// `tol = tol'/(4 - tol')`.
ds_sol ds_expl_crit3(ds_problem);


// Implementations in implicit.c.
/// Computes a diagonal scaling `P=diag(x)*A*diag(y)` of the nonnegative matrix A
/// with row and column sums approximately equal to r and c.
///
/// # Implementation
/// The implicit variant of the Sinkhorn--Knopp-like algorithm, proposed by
/// B. Kalantari and L. Khachiyan in [_On the complexity of nonnegative-matrix scaling_](https://core.ac.uk/download/pdf/82614641.pdf).
/// The main ideas are summarized in the paper [_The Sinkhorn-Knopp algorithm: convergence and applications_](https://strathprints.strath.ac.uk/19685/1/skapp.pdf)
/// by P. A. Knight.
///
/// # Stopping criterion
/// The iterative process stops when the `l_\infty` norm of the vector
/// `[x_k ./ x_{k-1}; y_k ./ y_{k-1}] - ones(m+n,1)` is less than or equal to
/// `tol`, or after `max_iters` iterations have been run.
ds_sol ds_impl_crit1(ds_problem);

/// Computes a diagonal scaling `P=diag(x)*A*diag(y)` of the nonnegative matrix A
/// with row and column sums approximately equal to r and c.
///
/// # Implementation
/// The implicit variant of the Sinkhorn--Knopp-like algorithm, proposed by
/// B. Kalantari and L. Khachiyan in [_On the complexity of nonnegative-matrix scaling_](https://core.ac.uk/download/pdf/82614641.pdf).
/// The main ideas are summarized in the paper [_The Sinkhorn-Knopp algorithm: convergence and applications_](https://strathprints.strath.ac.uk/19685/1/skapp.pdf)
/// by P. A. Knight.
///
/// # Stopping criterion
/// Let `v_k = [x_k ./ x_{k-1}; y_k ./ y_{k-1}]`. The iterative process stops when
/// the quotient of the largest entry of v_k and the smallest is less than or
/// equal to `(1+tol)/(1-tol)`, or after `max_iters` iterations have been run.
ds_sol ds_impl_crit2(ds_problem);

/// Computes a diagonal scaling `P=diag(x)*A*diag(y)` of the nonnegative matrix A
/// with row and column sums approximately equal to r and c.
///
/// # Implementation
/// The implicit variant of the Sinkhorn--Knopp-like algorithm, proposed by
/// B. Kalantari and L. Khachiyan in [_On the complexity of nonnegative-matrix scaling_](https://core.ac.uk/download/pdf/82614641.pdf).
/// The main ideas are summarized in the paper [_The Sinkhorn-Knopp algorithm: convergence and applications_](https://strathprints.strath.ac.uk/19685/1/skapp.pdf)
/// by P. A. Knight.
///
/// # Stopping criterion
/// Let `u_k = x_k ./ x_{k-1}` and `v_k = y_k ./ y_{k-1}`. The iterative
/// process stops when `max{max u_k / min u_k, max v_k / min v_k}` is less than
/// or equal to `(1+tol)/(1-tol)`, or after `max_iters` iterations have been run.
/// To use the same criterion as in [Diagonal scalings for the eigenstructure of arbitrary pencils](https://gauss.uc3m.es/fdopico/papers/simax2020.pdf)
/// by F. M. Dopico et al., set `tol = tol'/(4 - tol')`.
ds_sol ds_impl_crit3(ds_problem);


// Implementations in newton.c.
// Warning: the starting address of the solution vector y returned by the Newton's
// method-based diagonal scaling routines might not be a multiple of DS_VEC_BYTE_WIDTH
// bytes. This is required to avoid expensive copying after the solution of
// certain linear systems with LAPACK's `dgels` routine.

/// Computes a diagonal scaling `P=diag(x)*A*diag(y)` of the nonnegative matrix A
/// with row and column sums approximately equal to r and c.
///
/// # Implementation
/// A novel extension of P. A. Knight and D. Ruiz's symmetric matrix balancing method
/// proposed in Section 2 of [_A fast algorithm for matrix balancing_](https://d-nb.info/991914708/34)
/// to the general diagonal scaling problem. See Section 4.2 of H. Sanz González's [bachelor thesis](https://hgsg.me/bachelor/)
/// for further information.
///
/// **Warning:** The starting address of the solution vector `y` returned by
/// this function might not be a multiple of `DS_VEC_BYTE_WIDTH` bytes.
///
/// # Stopping criterion
/// The iterative process stops when the `l_\infty` norm of the vector
/// `[x_k ./ x_{k-1}; y_k ./ y_{k-1}] - ones(m+n,1)` is less than or equal to
/// `tol`, or after `max_iters` iterations have been run.
ds_sol ds_newt_crit1(ds_problem);

/// Computes a diagonal scaling `P=diag(x)*A*diag(y)` of the nonnegative matrix A
/// with row and column sums approximately equal to r and c.
///
/// # Implementation
/// A novel extension of P. A. Knight and D. Ruiz's symmetric matrix balancing method
/// proposed in Section 2 of [_A fast algorithm for matrix balancing_](https://d-nb.info/991914708/34)
/// to the general diagonal scaling problem. See Section 4.2 of H. Sanz González's [bachelor thesis](https://hgsg.me/bachelor/)
/// for further information.
///
/// **Warning:** The starting address of the solution vector `y` returned by
/// this function might not be a multiple of `DS_VEC_BYTE_WIDTH` bytes.
///
/// # Stopping criterion
/// Let `v_k = [x_k ./ x_{k-1}; y_k ./ y_{k-1}]`. The iterative process stops when
/// the quotient of the largest entry of v_k and the smallest is less than or
/// equal to `(1+tol)/(1-tol)`, or after `max_iters` iterations have been run.
ds_sol ds_newt_crit2(ds_problem);

/// Computes a diagonal scaling `P=diag(x)*A*diag(y)` of the nonnegative matrix A
/// with row and column sums approximately equal to r and c.
///
/// # Implementation
/// A novel extension of P. A. Knight and D. Ruiz's symmetric matrix balancing method
/// proposed in Section 2 of [_A fast algorithm for matrix balancing_](https://d-nb.info/991914708/34)
/// to the general diagonal scaling problem. See Section 4.2 of H. Sanz González's [bachelor thesis](https://hgsg.me/bachelor/)
/// for further information.
///
/// **Warning:** The starting address of the solution vector `y` returned by
/// this function might not be a multiple of `DS_VEC_BYTE_WIDTH` bytes.
///
/// # Stopping criterion
/// Let `u_k = x_k ./ x_{k-1}` and `v_k = y_k ./ y_{k-1}`. The iterative
/// process stops when `max{max u_k / min u_k, max v_k / min v_k}` is less than
/// or equal to `(1+tol)/(1-tol)`, or after `max_iters` iterations have been run.
/// To use the same criterion as in (F. M. Dopico et al., 2022), set
/// `tol = tol'/(4 - tol')`.
ds_sol ds_newt_crit3(ds_problem);

#ifdef __cplusplus
}
#endif
#endif //DIAGONALSCALINGS_DIAG_SCALS_H
