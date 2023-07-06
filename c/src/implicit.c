#include <diag_scals/diag_scals.h>
#include <diag_scals/common.h>


ds_sol ds_impl_crit1(ds_problem pr) {
    ds_sol sol = ds_sol_init(&pr);
    if (ds_sol_is_err(&sol)) return sol;

    // Stores the vectors x and y from the previous iteration, padded
    // to the nearest `DS_VEC_BYTE_WIDTH`-byte boundary. This allows us to
    // treat this array as two separate, unaliased ones whose starting
    // address is a multiple of `VEC_LANE_COUNT` bytes.
    double *prev = ds_work_area_get(DS_ROUND_UP_TO_VEC_LANE_COUNT(pr.n) + pr.m);
    DS_SOL_VERIFY(prev, sol);
    double *__restrict y_prev = prev,
           *__restrict x_prev = &prev[DS_ROUND_UP_TO_VEC_LANE_COUNT(pr.n)];
    y_prev = __builtin_assume_aligned(y_prev, DS_VEC_BYTE_WIDTH);
    x_prev = __builtin_assume_aligned(x_prev, DS_VEC_BYTE_WIDTH);

    // Initialize the previous iterates to unity. We could also copy
    // the entries of sol.x and sol.y, but that requires loads.
    for (size_t k = 0; k < DS_ROUND_UP_TO_VEC_LANE_COUNT(pr.n) + pr.m; ++k) prev[k] = 1.;

    for (; sol.iters <= pr.max_iters; ++sol.iters) {
        // Compute y <- c ./ (A' * x).
        ds_mat_t_vec_prod(pr.a, sol.x, sol.y, pr.m, pr.n); // y <- A' * x.
        size_t j = 0;
        for (; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d cs = _mm256_load_pd(&pr.c[j]);
            __m256d ys = _mm256_load_pd(&sol.y[j]);
            _mm256_store_pd(&sol.y[j], _mm256_div_pd(cs, ys));
        }
        for (; j < pr.n; ++j)
            sol.y[j] = pr.c[j] / sol.y[j];

        // Compute x <- r ./ (A * y).
        ds_mat_vec_prod(pr.a, sol.y, sol.x, pr.m, pr.n); // x <- A * y.
        size_t i = 0;
        for (; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); i += DS_VEC_LANE_COUNT) {
            __m256d rs = _mm256_load_pd(&pr.r[i]);
            __m256d xs = _mm256_load_pd(&sol.x[i]);
            _mm256_store_pd(&sol.x[i], _mm256_div_pd(rs, xs));
        }
        for (; i < pr.m; ++i)
            sol.x[i] = pr.r[i] / sol.x[i];

        // Calculate the residual in two parts.
        __m256d ress = _mm256_setzero_pd(),
                ones = _mm256_set1_pd(1.);
        for (j = 0; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d y_prevs = _mm256_load_pd(&y_prev[j]);
            __m256d ys = _mm256_load_pd(&sol.y[j]);
            __m256d diffs = _mm256_sub_pd(_mm256_div_pd(ys, y_prevs), ones);
            ress = _mm256_max_pd(ress, ds_mm256_abs_pd(diffs));
        }
        sol.res = 0.;
        for (; j < pr.n; ++j)
            sol.res = DS_MAX(sol.res, fabs(sol.y[j] / y_prev[j] - 1.));
        for (i = 0; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.m); i += DS_VEC_LANE_COUNT) {
            __m256d x_prevs = _mm256_load_pd(&x_prev[i]);
            __m256d xs = _mm256_load_pd(&sol.x[i]);
            __m256d diffs = _mm256_sub_pd(_mm256_div_pd(xs, x_prevs), ones);
            ress = _mm256_max_pd(ress, ds_mm256_abs_pd(diffs));
        }
        for (; i < pr.m; ++i)
            sol.res = DS_MAX(sol.res, fabs(sol.x[i] / x_prev[i] - 1.));

        if ((sol.res = DS_MAX(sol.res, ds_mm256_rmax_pd(ress))) <= pr.tol) break;

        // Cache the current iterates for the next iteration.
        memcpy(y_prev, sol.y, pr.n * sizeof(double));
        memcpy(x_prev, sol.x, pr.m * sizeof(double));
    }

    // Scale rows and columns of A in place to obtain P.
    ds_mat_scale_rows_cols(pr.a, sol.x, sol.y, pr.m, pr.n);
    sol.p = pr.a;
    return sol;
}

ds_sol ds_impl_crit2(ds_problem pr) {
    ds_sol sol = ds_sol_init(&pr);
    if (ds_sol_is_err(&sol)) return sol;

    // Stores the vectors x and y from the previous iteration, padded
    // to the nearest `DS_VEC_BYTE_WIDTH`-byte boundary. This allows us to
    // treat this array as two separate, unaliased ones whose starting
    // address is a multiple of `VEC_LANE_COUNT` bytes.
    double *prev = ds_work_area_get(DS_ROUND_UP_TO_VEC_LANE_COUNT(pr.n) + pr.m);
    DS_SOL_VERIFY(prev, sol);
    double *__restrict y_prev = prev,
           *__restrict x_prev = &prev[DS_ROUND_UP_TO_VEC_LANE_COUNT(pr.n)];
    y_prev = __builtin_assume_aligned(y_prev, DS_VEC_BYTE_WIDTH);
    x_prev = __builtin_assume_aligned(x_prev, DS_VEC_BYTE_WIDTH);

    // Initialize the previous iterates to unity. We could also copy
    // the entries of sol.x and sol.y, but that requires loads.
    for (size_t k = 0; k < DS_ROUND_UP_TO_VEC_LANE_COUNT(pr.n) + pr.m; ++k) prev[k] = 1.;

    for (; sol.iters <= pr.max_iters; ++sol.iters) {
        // Compute y <- c ./ (A' * x).
        ds_mat_t_vec_prod(pr.a, sol.x, sol.y, pr.m, pr.n); // y <- A' * x.
        size_t j = 0;
        for (; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d cs = _mm256_load_pd(&pr.c[j]);
            __m256d ys = _mm256_load_pd(&sol.y[j]);
            _mm256_store_pd(&sol.y[j], _mm256_div_pd(cs, ys));
        }
        for (; j < pr.n; ++j)
            sol.y[j] = pr.c[j] / sol.y[j];

        // Compute x <- r ./ (A * y).
        ds_mat_vec_prod(pr.a, sol.y, sol.x, pr.m, pr.n); // x <- A * y.
        size_t i = 0;
        for (; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); i += DS_VEC_LANE_COUNT) {
            __m256d rs = _mm256_load_pd(&pr.r[i]);
            __m256d xs = _mm256_load_pd(&sol.x[i]);
            _mm256_store_pd(&sol.x[i], _mm256_div_pd(rs, xs));
        }
        for (; i < pr.m; ++i)
            sol.x[i] = pr.r[i] / sol.x[i];

        // Calculate the residual in two parts.
        __m256d max_mults = _mm256_setzero_pd(),
                min_mults = _mm256_set1_pd(INFINITY);
        for (j = 0; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d ys = _mm256_load_pd(&sol.y[j]);
            __m256d y_prevs = _mm256_load_pd(&y_prev[j]);
            ys = _mm256_div_pd(ys, y_prevs);
            max_mults = _mm256_max_pd(max_mults, ys);
            min_mults = _mm256_min_pd(min_mults, ys);
        }
        double rem_max_mult = 0., rem_min_mult = INFINITY;
        for (; j < pr.n; ++j) {
            double mult = sol.y[j] / y_prev[j];
            rem_max_mult = DS_MAX(rem_max_mult, mult);
            rem_min_mult = DS_MIN(rem_min_mult, mult);
        }

        for (i = 0; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.m); i += DS_VEC_LANE_COUNT) {
            __m256d xs = _mm256_load_pd(&sol.x[i]);
            __m256d x_prevs = _mm256_load_pd(&x_prev[i]);
            xs = _mm256_div_pd(xs, x_prevs);
            max_mults = _mm256_max_pd(max_mults, xs);
            min_mults = _mm256_min_pd(min_mults, xs);
        }
        for (; i < pr.m; ++i) {
            double mult = sol.x[i] / x_prev[i];
            rem_max_mult = DS_MAX(rem_max_mult, mult);
            rem_min_mult = DS_MIN(rem_min_mult, mult);
        }

        // Reduce the residual lanes.
        rem_max_mult = DS_MAX(rem_max_mult, ds_mm256_rmax_pd(max_mults));
        rem_min_mult = DS_MIN(rem_min_mult, ds_mm256_rmin_pd(min_mults));
        if ((sol.res = rem_max_mult / rem_min_mult) <= (1 + pr.tol) / (1 - pr.tol)) break;

        // Cache the current iterates for the next iteration.
        memcpy(y_prev, sol.y, pr.n * sizeof(double));
        memcpy(x_prev, sol.x, pr.m * sizeof(double));
    }

    // Scale rows and columns of A in place to obtain P.
    ds_mat_scale_rows_cols(pr.a, sol.x, sol.y, pr.m, pr.n);
    sol.p = pr.a;
    return sol;
}

ds_sol ds_impl_crit3(ds_problem pr) {
    ds_sol sol = ds_sol_init(&pr);
    if (ds_sol_is_err(&sol)) return sol;

    // Stores the vectors x and y from the previous iteration, padded
    // to the nearest `DS_VEC_BYTE_WIDTH`-byte boundary. This allows us to
    // treat this array as two separate, unaliased ones whose starting
    // address is a multiple of `VEC_LANE_COUNT` bytes.
    double *prev = ds_work_area_get(DS_ROUND_UP_TO_VEC_LANE_COUNT(pr.n) + pr.m);
    DS_SOL_VERIFY(prev, sol);
    double *__restrict y_prev = prev,
           *__restrict x_prev = &prev[DS_ROUND_UP_TO_VEC_LANE_COUNT(pr.n)];
    y_prev = __builtin_assume_aligned(y_prev, DS_VEC_BYTE_WIDTH);
    x_prev = __builtin_assume_aligned(x_prev, DS_VEC_BYTE_WIDTH);

    // Initialize the previous iterates to unity. We could also copy
    // the entries of sol.x and sol.y, but that requires loads.
    for (size_t k = 0; k < DS_ROUND_UP_TO_VEC_LANE_COUNT(pr.n) + pr.m; ++k) prev[k] = 1.;

    for (; sol.iters <= pr.max_iters; ++sol.iters) {
        // Compute y <- c ./ (A' * x).
        ds_mat_t_vec_prod(pr.a, sol.x, sol.y, pr.m, pr.n); // y <- A' * x.
        size_t j = 0;
        for (; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d cs = _mm256_load_pd(&pr.c[j]);
            __m256d ys = _mm256_load_pd(&sol.y[j]);
            _mm256_store_pd(&sol.y[j], _mm256_div_pd(cs, ys));
        }
        for (; j < pr.n; ++j)
            sol.y[j] = pr.c[j] / sol.y[j];

        // Compute x <- r ./ (A * y).
        ds_mat_vec_prod(pr.a, sol.y, sol.x, pr.m, pr.n); // x <- A * y.
        size_t i = 0;
        for (; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); i += DS_VEC_LANE_COUNT) {
            __m256d rs = _mm256_load_pd(&pr.r[i]);
            __m256d xs = _mm256_load_pd(&sol.x[i]);
            _mm256_store_pd(&sol.x[i], _mm256_div_pd(rs, xs));
        }
        for (; i < pr.m; ++i)
            sol.x[i] = pr.r[i] / sol.x[i];

        // Calculate the residual in two parts.
        __m256d y_max_mults = _mm256_setzero_pd(),
                y_min_mults = _mm256_set1_pd(INFINITY);
        for (j = 0; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d ys = _mm256_load_pd(&sol.y[j]);
            __m256d y_prevs = _mm256_load_pd(&y_prev[j]);
            ys = _mm256_div_pd(ys, y_prevs);
            y_max_mults = _mm256_max_pd(y_max_mults, ys);
            y_min_mults = _mm256_min_pd(y_min_mults, ys);
        }
        double y_max_mult = ds_mm256_rmax_pd(y_max_mults),
                y_min_mult = ds_mm256_rmin_pd(y_min_mults);
        for (; j < pr.n; ++j) {
            double mult = sol.y[j] / y_prev[j];
            y_max_mult = DS_MAX(y_max_mult, mult);
            y_min_mult = DS_MIN(y_min_mult, mult);
        }

        __m256d x_max_mults = _mm256_setzero_pd(),
                x_min_mults = _mm256_set1_pd(INFINITY);
        for (i = 0; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.m); i += DS_VEC_LANE_COUNT) {
            __m256d xs = _mm256_load_pd(&sol.x[i]);
            __m256d x_prevs = _mm256_load_pd(&x_prev[i]);
            xs = _mm256_div_pd(xs, x_prevs);
            x_max_mults = _mm256_max_pd(x_max_mults, xs);
            x_min_mults = _mm256_min_pd(x_min_mults, xs);
        }
        double x_max_mult = ds_mm256_rmax_pd(x_max_mults),
                x_min_mult = ds_mm256_rmin_pd(x_min_mults);
        for (; i < pr.m; ++i) {
            double mult = sol.x[i] / x_prev[i];
            x_max_mult = DS_MAX(x_max_mult, mult);
            x_min_mult = DS_MIN(x_min_mult, mult);
        }

        // Reduce the residual lanes.
        sol.res = DS_MAX(x_max_mult / x_min_mult, y_max_mult / y_min_mult);
        if (sol.res <= (1 + pr.tol) / (1 - pr.tol)) break;

        // Cache the current iterates for the next iteration.
        memcpy(y_prev, sol.y, pr.n * sizeof(double));
        memcpy(x_prev, sol.x, pr.m * sizeof(double));
    }

    // Scale rows and columns of A in place to obtain P.
    ds_mat_scale_rows_cols(pr.a, sol.x, sol.y, pr.m, pr.n);
    sol.p = pr.a;
    return sol;
}
