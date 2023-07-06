#include <float.h>
#include <mkl_lapacke.h>
#include <diag_scals/diag_scals.h>
#include <diag_scals/common.h>


#define DS_NEWT_SOLVE_QR

// Similar to ds_sol_init, but guarantees that the vectors x and y
// are contiguous in memory. The starting address of `sol.x` is
// a multiple of `DS_VEC_BYTE_WIDTH`, but `sol.y` might be unaligned.
ds_sol ds_newt_sol_init(ds_problem *pr) {
    double *z = ds_vec_alloc(pr->m + pr->n + 1);
    if (!z) return (ds_sol){ .iters = DS_ERROR };
    ds_sol sol = {.x = z, .y = &z[pr->m], .iters = 1};
    // Set initial iterates.
    for (size_t i = 0; i < pr->m; ++i) sol.x[i] = 1. / (double) pr->m;
    for (size_t j = 0; j < pr->n; ++j) sol.y[j] = 1. / (double) pr->n;
    z[pr->m + pr->n] = 0.; // independent term of last equation of B[x;y] = z.
    return sol;
}

ds_sol ds_newt_crit1(ds_problem pr) {
    // This method relies on the least squares solution [x; y] to the linear
    // system B[x;y] = z. The coefficient matrix can be partitioned in six
    // blocks, where the two bottom blocks contain respectively 1s and -1s.

    // The array z is overwritten to contain the solution [x; y] of the system
    // by the DGELS LAPACK routine. Thus, we cannot guarantee that the subarray
    // y starting at address &z[m] is a multiple of DS_VEC_BYTE_WIDTH. We can
    // still treat the x and y arrays as two separate, unaliased ones, however.
    ds_sol sol = ds_newt_sol_init(&pr);
    if (ds_sol_is_err(&sol)) return sol;
    double *__restrict z = __builtin_assume_aligned(sol.x, DS_VEC_BYTE_WIDTH);

    // Request space to store the matrix B and the vectors x, y from the previous
    // iterations (padded to the nearest `DS_VEC_BYTE_WIDTH`-byte boundary).
    size_t b_m = pr.m + pr.n + 1, b_n = pr.m + pr.n; // dimensions of B.
    double *aux = ds_work_area_get(DS_ROUND_UP_TO_VEC_LANE_COUNT(b_m * b_n) + pr.m + pr.n); // B, x_prev, y_prev
    DS_SOL_VERIFY(aux, sol);
    double *__restrict b = __builtin_assume_aligned(aux, DS_VEC_BYTE_WIDTH),
            *__restrict prev =  &aux[DS_ROUND_UP_TO_VEC_LANE_COUNT(b_m * b_n)],
            *__restrict x_prev = prev, *__restrict y_prev = &prev[pr.m];
    prev = __builtin_assume_aligned(prev, DS_VEC_BYTE_WIDTH);
    x_prev = __builtin_assume_aligned(x_prev, DS_VEC_BYTE_WIDTH);

#ifndef DS_NEWT_SOLVE_QR
    lapack_int rank; double rcond = 1e-4;
    lapack_int *__restrict jpvt = calloc(b_n, sizeof(lapack_int));
    DS_SOL_VERIFY(jpvt, sol);
#endif

    for (; sol.iters <= pr.max_iters; ++sol.iters) {
        // Cache the previous iterates for the current iteration.
        // We need to do this right now, because z is aliased to
        // `sol.x` and `sol.y`.
        memcpy(prev, z, b_n * sizeof(double));

        // Set up the vector z. The top entries z[1:m] <- (A*y) .* x + r,
        // and we can copy the result of A*y to the main diagonal of the
        // top-left block of B during this process to save some computation.
        ds_mat_vec_prod(pr.a, sol.y /* or y_prev, doesn't matter */, z, pr.m, pr.n);
        ds_mat_sub_set_diag(b, z, pr.n + 1, pr.m); // B[1:m, 1:m] <- diag(A*y)
        size_t i = 0;
        for (; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.m); i += DS_VEC_LANE_COUNT) {
            __m256d zs = _mm256_load_pd(&z[i]);
            __m256d xs = _mm256_load_pd(&x_prev[i]);
            __m256d rs = _mm256_load_pd(&pr.r[i]);
            zs = _mm256_mul_pd(zs, xs);
            _mm256_store_pd(&z[i], _mm256_add_pd(zs, rs));
        }
        for (; i < pr.m; ++i) z[i] = z[i] * x_prev[i] + pr.r[i];

        // z[m+1:m+n] <- (A'*x) .* y + c.
        ds_mat_t_vec_prod(pr.a, x_prev, &z[pr.m], pr.m, pr.n);

        // Copy results of A'*x to main diagonal of the bottom-right nxn block of B,
        // excluding the row of -1s.
        ds_mat_sub_set_diag(&b[b_m * pr.m + pr.m], &z[pr.m], pr.m + 1, pr.n);
        size_t j = 0;
        for (; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d zs = _mm256_loadu_pd(&z[pr.m + j]);
            __m256d ys = _mm256_loadu_pd(&y_prev[j]);
            __m256d cs = _mm256_load_pd(&pr.c[j]);
            zs = _mm256_mul_pd(zs, ys);
            _mm256_storeu_pd(&z[pr.m + j], _mm256_add_pd(zs, cs));
        }
        for (; j < pr.n; ++j) z[pr.m + j] = z[pr.m + j] * y_prev[j] + pr.c[j];

        // Set up the antidiagonal blocks of B.
        // B[1:m, m+1:m+n] <- diag(x)*A.
        ds_mat_scale_rows_with_inc(pr.a, x_prev, &b[b_m * pr.m], pr.n + 1, pr.m, pr.n);
        // B[m+1:m+n, 1:m] <- diag(y)*A'.
        ds_mat_t_scale_rows_with_inc(pr.a, y_prev, &b[pr.m], pr.m + 1, pr.m, pr.n);

        // Fill last row of B with m 1's and n -1's. (Inefficient traversal by rows.)
        for (j = 0; j < pr.m; ++j) b[b_m * j + (b_m - 1)] = 1.;
        for (; j < pr.m + pr.n; ++j) b[b_m * j + (b_m - 1)] = -1.;

        // Solve the system B*w = z in the least squares sense.
#ifdef DS_NEWT_SOLVE_QR
        DS_SOL_VERIFY(!LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', b_m, b_n, 1, b, b_m, z, b_m), sol);
#else
        DS_SOL_VERIFY(!LAPACKE_dgelsy(LAPACK_COL_MAJOR, b_m, b_n, 1, b, b_m, z, b_m, jpvt, rcond, &rank), sol);
#endif
        // At this point sol.x and sol.y (aliased to z) contain the found solution.

        // Check that the solution is contained in the positive cone.
        __m256d cmp_res = _mm256_set1_pd(DBL_MIN); // broadcast machine epsilon.
        size_t k = 0;
        for (; k < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(b_n); k += DS_VEC_LANE_COUNT) {
            __m256d zs = _mm256_load_pd(&z[k]);
            cmp_res = _mm256_min_pd(cmp_res, zs);
        }
        DS_SOL_VERIFY(ds_mm256_rmin_pd(cmp_res) == DBL_MIN, sol);
        for (; k < b_n; ++k)
            DS_SOL_VERIFY(z[k] > 0., sol);

        // Compute the residual and evaluate the stopping criterion.
        // Optimization: the vectors x, y and x_prev, y_prev are contiguous,
        // so we can compute the l_\infty norm in one go.
        __m256d ress = _mm256_setzero_pd(), ones = _mm256_set1_pd(1.);
        for (k = 0; k < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(b_n); k += DS_VEC_LANE_COUNT) {
            __m256d prevs = _mm256_load_pd(&prev[k]);
            __m256d currs = _mm256_load_pd(&z[k]);
            __m256d diffs = _mm256_sub_pd(_mm256_div_pd(currs, prevs), ones);
            ress = _mm256_max_pd(ress, ds_mm256_abs_pd(diffs));
        }
        sol.res = ds_mm256_rmax_pd(ress);
        for (; k < b_n; ++k) sol.res = DS_MAX(sol.res, fabs(z[k] / prev[k] - 1.));

        if (sol.res <= pr.tol) break;
#ifndef DS_NEWT_SOLVE_QR
        // Clear old permutation.
        memset(jpvt, 0, b_n * sizeof(lapack_int));
#endif
    }
    // Scale rows and columns of A in place to obtain P.
    ds_mat_scale_rows_cols(pr.a, sol.x, sol.y, pr.m, pr.n);
    sol.p = pr.a;
#ifndef DS_NEWT_SOLVE_QR
    free(jpvt);
#endif
    return sol;
}

ds_sol ds_newt_crit2(ds_problem pr) {
    ds_sol sol = ds_newt_sol_init(&pr);
    if (ds_sol_is_err(&sol)) return sol;

    // The array z is overwritten to contain the solution [x; y] of the system
    // by the DGELS LAPACK routine. Thus, we cannot guarantee that the subarray
    // y starting at address &z[m] is a multiple of DS_VEC_BYTE_WIDTH. We can
    // still treat the x and y arrays as two separate, unaliased ones, however.
    double *__restrict z = __builtin_assume_aligned(sol.x, DS_VEC_BYTE_WIDTH);

    // Request space to store the matrix B and the vectors x, y from the previous
    // iterations (padded to the nearest `DS_VEC_BYTE_WIDTH`-byte boundary).
    size_t b_m = pr.m + pr.n + 1, b_n = pr.m + pr.n; // dimensions of B.
    double *aux = ds_work_area_get(DS_ROUND_UP_TO_VEC_LANE_COUNT(b_m * b_n) + pr.m + pr.n); // B, x_prev, y_prev
    DS_SOL_VERIFY(aux, sol);
    double *__restrict b = __builtin_assume_aligned(aux, DS_VEC_BYTE_WIDTH),
            *__restrict prev =  &aux[DS_ROUND_UP_TO_VEC_LANE_COUNT(b_m * b_n)],
            *__restrict x_prev = prev, *__restrict y_prev = &prev[pr.m];
    prev = __builtin_assume_aligned(prev, DS_VEC_BYTE_WIDTH);
    x_prev = __builtin_assume_aligned(x_prev, DS_VEC_BYTE_WIDTH);

#ifndef DS_NEWT_SOLVE_QR
    lapack_int rank; double rcond = 1e-4;
    lapack_int *__restrict jpvt = calloc(b_n, sizeof(lapack_int));
    DS_SOL_VERIFY(jpvt, sol);
#endif

    for (; sol.iters <= pr.max_iters; ++sol.iters) {
        // Cache the previous iterates for the current iteration.
        // We need to do this right now, because z is aliased to
        // `sol.x` and `sol.y`.
        memcpy(prev, z, b_n * sizeof(double));

        // Set up the vector z. The top entries z[1:m] <- (A*y) .* x + r,
        // and we can copy the result of A*y to the main diagonal of the
        // top-left block of B during this process to save some computation.
        ds_mat_vec_prod(pr.a, sol.y /* or y_prev, doesn't matter */, z, pr.m, pr.n);
        ds_mat_sub_set_diag(b, z, pr.n + 1, pr.m); // B[1:m, 1:m] <- diag(A*y)
        size_t i = 0;
        for (; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.m); i += DS_VEC_LANE_COUNT) {
            __m256d zs = _mm256_load_pd(&z[i]);
            __m256d xs = _mm256_load_pd(&x_prev[i]);
            __m256d rs = _mm256_load_pd(&pr.r[i]);
            zs = _mm256_mul_pd(zs, xs);
            _mm256_store_pd(&z[i], _mm256_add_pd(zs, rs));
        }
        for (; i < pr.m; ++i) z[i] = z[i] * x_prev[i] + pr.r[i];

        // z[m+1:m+n] <- (A'*x) .* y + c.
        ds_mat_t_vec_prod(pr.a, x_prev, &z[pr.m], pr.m, pr.n);

        // Copy results of A'*x to main diagonal of the bottom-right nxn block of B,
        // excluding the row of -1s.
        ds_mat_sub_set_diag(&b[b_m * pr.m + pr.m], &z[pr.m], pr.m + 1, pr.n);
        size_t j = 0;
        for (; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d zs = _mm256_loadu_pd(&z[pr.m + j]);
            __m256d ys = _mm256_loadu_pd(&y_prev[j]);
            __m256d cs = _mm256_load_pd(&pr.c[j]);
            zs = _mm256_mul_pd(zs, ys);
            _mm256_storeu_pd(&z[pr.m + j], _mm256_add_pd(zs, cs));
        }
        for (; j < pr.n; ++j) z[pr.m + j] = z[pr.m + j] * y_prev[j] + pr.c[j];

        // Set up the antidiagonal blocks of B.
        // B[1:m, m+1:m+n] <- diag(x)*A.
        ds_mat_scale_rows_with_inc(pr.a, x_prev, &b[b_m * pr.m], pr.n + 1, pr.m, pr.n);
        // B[m+1:m+n, 1:m] <- diag(y)*A'.
        ds_mat_t_scale_rows_with_inc(pr.a, y_prev, &b[pr.m], pr.m + 1, pr.m, pr.n);

        // Fill last row of B with m 1's and n -1's. (Inefficient traversal by rows.)
        for (j = 0; j < pr.m; ++j) b[b_m * j + (b_m - 1)] = 1.;
        for (; j < pr.m + pr.n; ++j) b[b_m * j + (b_m - 1)] = -1.;

        // Solve the system B*w = z in the least squares sense.
#ifdef DS_NEWT_SOLVE_QR
        DS_SOL_VERIFY(!LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', b_m, b_n, 1, b, b_m, z, b_m), sol);
#else
        DS_SOL_VERIFY(!LAPACKE_dgelsy(LAPACK_COL_MAJOR, b_m, b_n, 1, b, b_m, z, b_m, jpvt, rcond, &rank), sol);
#endif
        // At this point sol.x and sol.y (aliased to z) contain the found solution.

        // Check that the solution is contained in the positive cone.
        __m256d cmp_res = _mm256_set1_pd(DBL_MIN); // broadcast machine epsilon.
        size_t k = 0;
        for (; k < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(b_n); k += DS_VEC_LANE_COUNT) {
            __m256d zs = _mm256_load_pd(&z[k]);
            cmp_res = _mm256_min_pd(cmp_res, zs);
        }
        DS_SOL_VERIFY(ds_mm256_rmin_pd(cmp_res) == DBL_MIN, sol);
        for (; k < b_n; ++k)
            DS_SOL_VERIFY(z[k] > 0., sol);

        // Compute the residual and evaluate the stopping criterion.
        // Optimization: the vectors x, y and x_prev, y_prev are contiguous,
        // so we can compute the spectral condition number in one pass.
        __m256d max_mults = _mm256_setzero_pd(),
                min_mults = _mm256_set1_pd(INFINITY);
        for (k = 0; k < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(b_n); k += DS_VEC_LANE_COUNT) {
            __m256d prevs = _mm256_load_pd(&prev[k]);
            __m256d mults = _mm256_div_pd(_mm256_load_pd(&z[k]), prevs);
            max_mults = _mm256_max_pd(max_mults, mults);
            min_mults = _mm256_min_pd(min_mults, mults);
        }
        // Use two additional variables to keep track of the maximum and minimum
        // factors that lie in the last incomplete block.
        double rem_max_mult = 0., rem_min_mult = INFINITY;
        for (; k < b_n; ++k) {
            double mult = z[k] / prev[k];
            rem_max_mult = DS_MAX(rem_max_mult, mult);
            rem_min_mult = DS_MAX(rem_min_mult, mult);
        }

        // Reduce the residual lanes.
        rem_max_mult = DS_MAX(rem_max_mult, ds_mm256_rmax_pd(max_mults));
        rem_min_mult = DS_MIN(rem_min_mult, ds_mm256_rmin_pd(min_mults));
        if ((sol.res = rem_max_mult / rem_min_mult) <= (1 + pr.tol) / (1 - pr.tol)) break;
#ifndef DS_NEWT_SOLVE_QR
        // Clear old permutation.
        memset(jpvt, 0, b_n * sizeof(lapack_int));
#endif
    }
    // Scale rows and columns of A in place to obtain P.
    ds_mat_scale_rows_cols(pr.a, sol.x, sol.y, pr.m, pr.n);
    sol.p = pr.a;
#ifndef DS_NEWT_SOLVE_QR
    free(jpvt);
#endif
    return sol;
}

ds_sol ds_newt_crit3(ds_problem pr) {
    // The array z is overwritten to contain the solution [x; y] of the system
    // by the DGELS LAPACK routine. Thus, we cannot guarantee that the subarray
    // y starting at address &z[m] is a multiple of DS_VEC_BYTE_WIDTH. We can
    // still treat the x and y arrays as two separate, unaliased ones, however.
    ds_sol sol = ds_newt_sol_init(&pr);
    if (ds_sol_is_err(&sol)) return sol;
    double *__restrict z = __builtin_assume_aligned(sol.x, DS_VEC_BYTE_WIDTH);

    // Request space to store the matrix B and the vectors x, y from the previous
    // iterations (padded to the nearest `DS_VEC_BYTE_WIDTH`-byte boundary).
    size_t b_m = pr.m + pr.n + 1, b_n = pr.m + pr.n; // dimensions of B.
    double *aux = ds_work_area_get(DS_ROUND_UP_TO_VEC_LANE_COUNT(b_m * b_n) + pr.m + pr.n); // B, x_prev, y_prev
    DS_SOL_VERIFY(aux, sol);
    double *__restrict b = __builtin_assume_aligned(aux, DS_VEC_BYTE_WIDTH),
            *__restrict prev =  &aux[DS_ROUND_UP_TO_VEC_LANE_COUNT(b_m * b_n)],
            *__restrict x_prev = prev, *__restrict y_prev = &prev[pr.m];
    prev = __builtin_assume_aligned(prev, DS_VEC_BYTE_WIDTH);
    x_prev = __builtin_assume_aligned(x_prev, DS_VEC_BYTE_WIDTH);

#ifndef DS_NEWT_SOLVE_QR
    lapack_int rank; double rcond = 1e-4;
    lapack_int *__restrict jpvt = calloc(b_n, sizeof(lapack_int));
    DS_SOL_VERIFY(jpvt, sol);
#endif

    for (; sol.iters <= pr.max_iters; ++sol.iters) {
        // Cache the previous iterates for the current iteration.
        // We need to do this right now, because z is aliased to
        // `sol.x` and `sol.y`.
        memcpy(prev, z, b_n * sizeof(double));

        // Set up the vector z. The top entries z[1:m] <- (A*y) .* x + r,
        // and we can copy the result of A*y to the main diagonal of the
        // top-left block of B during this process to save some computation.
        ds_mat_vec_prod(pr.a, sol.y /* or y_prev, doesn't matter */, z, pr.m, pr.n);
        ds_mat_sub_set_diag(b, z, pr.n + 1, pr.m); // B[1:m, 1:m] <- diag(A*y)
        size_t i = 0;
        for (; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.m); i += DS_VEC_LANE_COUNT) {
            __m256d zs = _mm256_load_pd(&z[i]);
            __m256d xs = _mm256_load_pd(&x_prev[i]);
            __m256d rs = _mm256_load_pd(&pr.r[i]);
            zs = _mm256_mul_pd(zs, xs);
            _mm256_store_pd(&z[i], _mm256_add_pd(zs, rs));
        }
        for (; i < pr.m; ++i) z[i] = z[i] * x_prev[i] + pr.r[i];

        // z[m+1:m+n] <- (A'*x) .* y + c.
        ds_mat_t_vec_prod(pr.a, x_prev, &z[pr.m], pr.m, pr.n);

        // Copy results of A'*x to main diagonal of the bottom-right nxn block of B,
        // excluding the row of -1s.
        ds_mat_sub_set_diag(&b[b_m * pr.m + pr.m], &z[pr.m], pr.m + 1, pr.n);
        size_t j = 0;
        for (; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d zs = _mm256_loadu_pd(&z[pr.m + j]);
            __m256d ys = _mm256_loadu_pd(&y_prev[j]);
            __m256d cs = _mm256_load_pd(&pr.c[j]);
            zs = _mm256_mul_pd(zs, ys);
            _mm256_storeu_pd(&z[pr.m + j], _mm256_add_pd(zs, cs));
        }
        for (; j < pr.n; ++j) z[pr.m + j] = z[pr.m + j] * y_prev[j] + pr.c[j];

        // Set up the antidiagonal blocks of B.
        // B[1:m, m+1:m+n] <- diag(x)*A.
        ds_mat_scale_rows_with_inc(pr.a, x_prev, &b[b_m * pr.m], pr.n + 1, pr.m, pr.n);
        // B[m+1:m+n, 1:m] <- diag(y)*A'.
        ds_mat_t_scale_rows_with_inc(pr.a, y_prev, &b[pr.m], pr.m + 1, pr.m, pr.n);

        // Fill last row of B with m 1's and n -1's. (Inefficient traversal by rows.)
        for (j = 0; j < pr.m; ++j) b[b_m * j + (b_m - 1)] = 1.;
        for (; j < pr.m + pr.n; ++j) b[b_m * j + (b_m - 1)] = -1.;

        // Solve the system B*w = z in the least squares sense.
#ifdef DS_NEWT_SOLVE_QR
        DS_SOL_VERIFY(!LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', b_m, b_n, 1, b, b_m, z, b_m), sol);
#else
        DS_SOL_VERIFY(!LAPACKE_dgelsy(LAPACK_COL_MAJOR, b_m, b_n, 1, b, b_m, z, b_m, jpvt, rcond, &rank), sol);
#endif
        // At this point sol.x and sol.y (aliased to z) contain the found solution.

        // Check that the solution is contained in the positive cone.
        __m256d cmp_res = _mm256_set1_pd(DBL_MIN); // broadcast machine epsilon.
        size_t k = 0;
        for (; k < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(b_n); k += DS_VEC_LANE_COUNT) {
            __m256d zs = _mm256_load_pd(&z[k]);
            cmp_res = _mm256_min_pd(cmp_res, zs);
        }
        DS_SOL_VERIFY(ds_mm256_rmin_pd(cmp_res) == DBL_MIN, sol);
        for (; k < b_n; ++k)
            DS_SOL_VERIFY(z[k] > 0., sol);

        // Compute the residual and evaluate the stopping criterion.
        __m256d x_max_mults = _mm256_setzero_pd(),
                x_min_mults = _mm256_set1_pd(INFINITY);
        for (i = 0; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.m); i += DS_VEC_LANE_COUNT) {
            __m256d x_prevs = _mm256_load_pd(&x_prev[i]);
            __m256d x_mults = _mm256_div_pd(_mm256_load_pd(&sol.x[i]), x_prevs);
            x_max_mults = _mm256_max_pd(x_max_mults, x_mults);
            x_min_mults = _mm256_min_pd(x_min_mults, x_mults);
        }
        double x_max_mult = ds_mm256_rmax_pd(x_max_mults),
               x_min_mult = ds_mm256_rmin_pd(x_min_mults);
        for (; i < pr.m; ++i) {
            double x_mult = sol.x[i] / x_prev[i];
            x_max_mult = DS_MAX(x_max_mult, x_mult);
            x_min_mult = DS_MIN(x_min_mult, x_mult);
        }

        __m256d y_max_mults = _mm256_setzero_pd(),
                y_min_mults = _mm256_set1_pd(INFINITY);
        for (j = 0; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d y_prevs = _mm256_load_pd(&y_prev[j]);
            // `sol.y` is contiguous to `sol.x`, so it is not aligned to
            // the nearest `DS_VEC_LANE_COUNT`-byte boundary.
            __m256d y_mults = _mm256_div_pd(_mm256_loadu_pd(&sol.y[j]), y_prevs);
            y_max_mults = _mm256_max_pd(y_max_mults, y_mults);
            y_min_mults = _mm256_min_pd(y_min_mults, y_mults);
        }
        double y_max_mult = ds_mm256_rmax_pd(y_max_mults),
               y_min_mult = ds_mm256_rmin_pd(y_min_mults);
        for (; j < pr.n; ++j) {
            double y_mult = sol.y[j] / y_prev[j];
            y_max_mult = DS_MAX(y_max_mult, y_mult);
            y_min_mult = DS_MIN(y_min_mult, y_mult);
        }

        // Reduce the residual lanes.
        sol.res = DS_MAX(x_max_mult / x_min_mult, y_max_mult / y_min_mult);
        if (sol.res <= (1 + pr.tol) / (1 - pr.tol)) break;
#ifndef DS_NEWT_SOLVE_QR
        // Clear old permutation.
        memset(jpvt, 0, b_n * sizeof(lapack_int));
#endif
    }
    // Scale rows and columns of A in place to obtain P.
    ds_mat_scale_rows_cols(pr.a, sol.x, sol.y, pr.m, pr.n);
    sol.p = pr.a;
#ifndef DS_NEWT_SOLVE_QR
    free(jpvt);
#endif
    return sol;
}
