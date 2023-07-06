#include <diag_scals/diag_scals.h>
#include <diag_scals/common.h>


ds_sol ds_expl_crit1(ds_problem pr) {
    ds_sol sol = ds_sol_init(&pr);
    if (ds_sol_is_err(&sol)) return sol;

    double *__restrict mult = ds_work_area_get(DS_MAX(pr.m, pr.n));
    DS_SOL_VERIFY(mult, sol);
    mult = __builtin_assume_aligned(mult, DS_VEC_BYTE_WIDTH);
    for (; sol.iters <= pr.max_iters; ++sol.iters) {
        // Set y_mult <- c ./ csums(A), where y_mult is the nx1 subvector of mult.
        ds_mat_col_sums(pr.a, mult, pr.m, pr.n);
        size_t j = 0;
        for (; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d col_sums = _mm256_load_pd(&mult[j]);
            __m256d cs = _mm256_load_pd(&pr.c[j]);
            _mm256_store_pd(&mult[j], _mm256_div_pd(cs, col_sums));
        }
        for (; j < pr.n; ++j)
            mult[j] = pr.c[j] / mult[j];

        ds_mat_scale_cols(pr.a, mult, pr.m, pr.n); // A <- A .* y_mult.
        // The MATLAB implementation requires storing the previous y iterates
        // in order to calculate the elementwise quotient y ./ y_prev at
        // the end of this iteration for the residual. We can instead compute
        // this quantity here at the same time that we set y <- y .* y_mult.
        __m256d ress = _mm256_setzero_pd(),
                ones = _mm256_set1_pd(1.);
        for (j = 0; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d y_prev = _mm256_load_pd(&sol.y[j]);
            __m256d factors = _mm256_load_pd(&mult[j]);
            // Part of y <- y .* y_mult.
            _mm256_store_pd(&sol.y[j], _mm256_mul_pd(y_prev, factors));

            // The l_\infty norm of a vector is the maximum of its entries in
            // absolute value. Here we compute ||y_mult - 1||_\infty, leaving
            // the results across the lanes of y_ress.
            ress = _mm256_max_pd(ress,
                                 ds_mm256_abs_pd(_mm256_sub_pd(factors, ones)));
        }
        sol.res = 0.;
        for (; j < pr.n; ++j) {
            sol.y[j] *= mult[j];
            sol.res = DS_MAX(sol.res, fabs(mult[j] - 1.));
        }

        // Set x_mult <- r ./ rsums(A), where x_mult is the mx1 subvector of mult.
        ds_mat_row_sums(pr.a, mult, pr.m, pr.n);
        size_t i = 0;
        for (; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.m); i += DS_VEC_LANE_COUNT) {
            __m256d row_sums = _mm256_load_pd(&mult[i]);
            __m256d rs = _mm256_load_pd(&pr.r[i]);
            _mm256_store_pd(&mult[i], _mm256_div_pd(rs, row_sums));
        }
        for (; i < pr.m; ++i)
            mult[i] = pr.r[i] / mult[i];

        ds_mat_scale_rows(pr.a, mult, pr.m, pr.n); // A <- x_mult .* A
        // Again, we can perform the operation x <- x .* x_mult and the remaining
        // calculation of the residual simultaneously.
        for (i = 0; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.m); i += DS_VEC_LANE_COUNT) {
            __m256d x_prev = _mm256_load_pd(&sol.x[i]);
            __m256d factors = _mm256_load_pd(&mult[i]);
            // Part of x <- x .* x_mult.
            _mm256_store_pd(&sol.x[i], _mm256_mul_pd(x_prev, factors));

            // Find ||x_mult - 1||_\infty, accumulating the results into
            // the equivalent norm already found for the columns.
            ress = _mm256_max_pd(ress,
                                 ds_mm256_abs_pd(_mm256_sub_pd(factors, ones)));
        }
        for (; i < pr.m; ++i) {
            sol.x[i] *= mult[i];
            sol.res = DS_MAX(sol.res, fabs(mult[i] - 1.));
        }

        // Reduce the residual lanes.
        if ((sol.res = DS_MAX(sol.res, ds_mm256_rmax_pd(ress))) <= pr.tol) break;
    }

    sol.p = pr.a;
    return sol;
}

ds_sol ds_expl_crit2(ds_problem pr) {
    ds_sol sol = ds_sol_init(&pr);
    if (ds_sol_is_err(&sol)) return sol;

    double *__restrict mult = ds_work_area_get(DS_MAX(pr.m, pr.n));
    DS_SOL_VERIFY(mult, sol);
    mult = __builtin_assume_aligned(mult, DS_VEC_BYTE_WIDTH);
    for (; sol.iters <= pr.max_iters; ++sol.iters) {
        // Set y_mult <- c ./ csums(A), where y_mult is the nx1 subv.ector of mult.
        ds_mat_col_sums(pr.a, mult, pr.m, pr.n);
        size_t j = 0;
        for (; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d col_sums = _mm256_load_pd(&mult[j]);
            __m256d cs = _mm256_load_pd(&pr.c[j]);
            _mm256_store_pd(&mult[j], _mm256_div_pd(cs, col_sums));
        }
        for (; j < pr.n; ++j)
            mult[j] = pr.c[j] / mult[j];

        ds_mat_scale_cols(pr.a, mult, pr.m, pr.n); // A <- A .* y_mult.

        // The main difference between the explicit variant with stopping
        // criterion 1 and this one is that the residual now involves the
        // maximum *and* the minimum of x_mult and y_mult. The solution is
        // to keep two vector registers instead of a single one.
        // Thus, the previous optimization where the elementwise quotient
        // y ./ y_prev and the calculation of the residual are interspersed
        // still applies.
        __m256d max_mults = _mm256_setzero_pd(),
                min_mults = _mm256_set1_pd(INFINITY);
        for (j = 0; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d y_prev = _mm256_load_pd(&sol.y[j]);
            __m256d factors = _mm256_load_pd(&mult[j]);
            // Part of y <- y .* y_mult.
            _mm256_store_pd(&sol.y[j], _mm256_mul_pd(y_prev, factors));

            max_mults = _mm256_max_pd(max_mults, factors);
            min_mults = _mm256_min_pd(min_mults, factors);
        }
        // Use two additional variables to keep track of the maximum
        // and minimum factors that lie in the last incomplete block.
        double rem_max_mult = 0., rem_min_mult = INFINITY;
        for (; j < pr.n; ++j) {
            sol.y[j] *= mult[j];
            rem_max_mult = DS_MAX(rem_max_mult, mult[j]);
            rem_min_mult = DS_MIN(rem_min_mult, mult[j]);
        }

        // Set x_mult <- r ./ rsums(A), where x_mult is the mx1 subvector of mult.
        ds_mat_row_sums(pr.a, mult, pr.m, pr.n);
        size_t i = 0;
        for (; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.m); i += DS_VEC_LANE_COUNT) {
            __m256d row_sums = _mm256_load_pd(&mult[i]);
            __m256d rs = _mm256_load_pd(&pr.r[i]);
            _mm256_store_pd(&mult[i], _mm256_div_pd(rs, row_sums));
        }
        for (; i < pr.m; ++i)
            mult[i] = pr.r[i] / mult[i];

        ds_mat_scale_rows(pr.a, mult, pr.m, pr.n); // A <- x_mult .* A
        // Again, we can perform the operation x <- x .* x_mult and the remaining
        // calculation of the residual simultaneously.
        for (i = 0; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.m); i += DS_VEC_LANE_COUNT) {
            __m256d x_prev = _mm256_load_pd(&sol.x[i]);
            __m256d factors = _mm256_load_pd(&mult[i]);
            // Part of x <- x .* x_mult.
            _mm256_store_pd(&sol.x[i], _mm256_mul_pd(x_prev, factors));

            max_mults = _mm256_max_pd(max_mults, factors);
            min_mults = _mm256_min_pd(min_mults, factors);
        }
        for (; i < pr.m; ++i) {
            sol.x[i] *= mult[i];
            rem_max_mult = DS_MAX(rem_max_mult, mult[i]);
            rem_min_mult = DS_MIN(rem_min_mult, mult[i]);
        }

        // Reduce the residual lanes.
        rem_max_mult = DS_MAX(rem_max_mult, ds_mm256_rmax_pd(max_mults));
        rem_min_mult = DS_MIN(rem_min_mult, ds_mm256_rmin_pd(min_mults));
        if ((sol.res = rem_max_mult / rem_min_mult) <= (1 + pr.tol) / (1 - pr.tol)) break;
    }

    sol.p = pr.a;
    return sol;
}

ds_sol ds_expl_crit3(ds_problem pr) {
    ds_sol sol = ds_sol_init(&pr);
    if (ds_sol_is_err(&sol)) return sol;

    double *__restrict mult = ds_work_area_get(DS_MAX(pr.m, pr.n));
    DS_SOL_VERIFY(mult, sol);
    mult = __builtin_assume_aligned(mult, DS_VEC_BYTE_WIDTH);
    for (; sol.iters <= pr.max_iters; ++sol.iters) {
        // Set y_mult <- c ./ csums(A), where y_mult is the nx1 subvector of mult.

        ds_mat_col_sums(pr.a, mult, pr.m, pr.n);
        size_t j = 0;
        for (; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d col_sums = _mm256_load_pd(&mult[j]);
            __m256d cs = _mm256_load_pd(&pr.c[j]);
            _mm256_store_pd(&mult[j], _mm256_div_pd(cs, col_sums));
        }
        for (; j < pr.n; ++j)
            mult[j] = pr.c[j] / mult[j];

        ds_mat_scale_cols(pr.a, mult, pr.m, pr.n); // A <- A .* y_mult.

        // As in the explicit method with the second stopping criterion,
        // keep accumulator registers for the minimum and maximum.
        // The main difference is that now we calculate the minimum and
        // maximum for y_mults and x_mults separately.
        __m256d y_max_mults = _mm256_setzero_pd(),
                y_min_mults = _mm256_set1_pd(INFINITY);
        for (j = 0; j < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.n); j += DS_VEC_LANE_COUNT) {
            __m256d y_prev = _mm256_load_pd(&sol.y[j]);
            __m256d factors = _mm256_load_pd(&mult[j]);
            // Part of y <- y .* y_mult.
            _mm256_store_pd(&sol.y[j], _mm256_mul_pd(y_prev, factors));

            y_max_mults = _mm256_max_pd(y_max_mults, factors);
            y_min_mults = _mm256_min_pd(y_min_mults, factors);
        }
        // Reduce the lanes of the minimum and maximum accumulators,
        // and consider the entries in the last incomplete block.
        double y_max_mult = ds_mm256_rmax_pd(y_max_mults),
                y_min_mult = ds_mm256_rmin_pd(y_min_mults);
        for (; j < pr.n; ++j) {
            sol.y[j] *= mult[j];
            y_max_mult = DS_MAX(y_max_mult, mult[j]);
            y_min_mult = DS_MIN(y_min_mult, mult[j]);
        }

        // Set x_mult <- r ./ rsums(A), where x_mult is the mx1 subvector of mult.
        ds_mat_row_sums(pr.a, mult, pr.m, pr.n);
        size_t i = 0;
        for (; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.m); i += DS_VEC_LANE_COUNT) {
            __m256d row_sums = _mm256_load_pd(&mult[i]);
            __m256d rs = _mm256_load_pd(&pr.r[i]);
            _mm256_store_pd(&mult[i], _mm256_div_pd(rs, row_sums));
        }
        for (; i < pr.m; ++i)
            mult[i] = pr.r[i] / mult[i];

        ds_mat_scale_rows(pr.a, mult, pr.m, pr.n); // A <- x_mult .* A
        // Again, we can perform the operation x <- x .* x_mult and the remaining
        // calculation of the residual simultaneously.
        __m256d x_max_mults = _mm256_setzero_pd(),
                x_min_mults = _mm256_set1_pd(INFINITY);
        for (i = 0; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(pr.m); i += DS_VEC_LANE_COUNT) {
            __m256d x_prev = _mm256_load_pd(&sol.x[i]);
            __m256d factors = _mm256_load_pd(&mult[i]);
            // Part of x <- x .* x_mult.
            _mm256_store_pd(&sol.x[i], _mm256_mul_pd(x_prev, factors));

            x_max_mults = _mm256_max_pd(x_max_mults, factors);
            x_min_mults = _mm256_min_pd(x_min_mults, factors);
        }
        double x_max_mult = ds_mm256_rmax_pd(x_max_mults),
                x_min_mult = ds_mm256_rmin_pd(x_min_mults);
        for (; i < pr.m; ++i) {
            sol.x[i] *= mult[i];
            x_max_mult = DS_MAX(x_max_mult, mult[i]);
            x_min_mult = DS_MIN(x_min_mult, mult[i]);
        }

        // Reduce the residual lanes.
        sol.res = DS_MAX(y_max_mult / y_min_mult, x_max_mult / x_min_mult);
        if (sol.res <= (1 + pr.tol) / (1 - pr.tol)) break;
    }

    sol.p = pr.a;
    return sol;
}
