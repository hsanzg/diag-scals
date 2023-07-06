#include <stdio.h>
#include <diag_scals/common.h>
#include <diag_scals/diag_scals.h>


// Utilities for comparison of test objects.

#define DS_VERIFY(expr, msg, ...) if (__builtin_expect(!(expr), 1)) {printf((msg), ##__VA_ARGS__); fflush(0); exit(1);}

int nearly_equal_within_tol(const double a, const double b, const double fp_tol) {
    return fabs(a - b) <= fp_tol;
}

int nearly_equal(const double a, const double b) {
    return nearly_equal_within_tol(a, b, 1e-10);
}

void verify_vecs_equal(const double *u, const double *v, const size_t n) {
    for (size_t i = 0; i < n; ++i)
        DS_VERIFY(nearly_equal(u[i], v[i]), "found not nearly equal values:"
                                            "u[%lu] = %lf, but v[%lu] = %lf",
                  i, u[i], i, v[i]);
}

void verify_mats_equal(const double *a, const double *b,
                       const size_t m, const size_t n) {
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            DS_VERIFY(nearly_equal(a[m * j + i], b[m * j + i]), "found not nearly equal values:"
                                                                "a[%lu, %lu] = %lf, but b[%lu, %lu] = %lf",
                      i, j, a[m * j + i], i, j, b[m * j + i]);
}

void verify_diag_scal_errs(const ds_problem *pr, const ds_sol *sol, const double err_tol) {
    DS_VERIFY(!ds_sol_is_err(sol), "sol is erroneous");
    // Compute row and column sums of diagonal scaling.
    double *rs = ds_vec_alloc(pr->m),
           *cs = ds_vec_alloc(pr->n);
    DS_VERIFY(rs, "rs alloc failed"); DS_VERIFY(cs, "cs alloc failed");

    ds_mat_row_sums(sol->p, rs, pr->m, pr->n);
    for (size_t i = 0; i < pr->m; ++i)
        DS_VERIFY(nearly_equal_within_tol(pr->r[i], rs[i], err_tol), "%luth row sum; got %e, expected %e",
                  i, rs[i], pr->r[i]);
    free(rs);

    ds_mat_col_sums(sol->p, cs, pr->m, pr->n);
    for (size_t j = 0; j < pr->n; ++j)
        DS_VERIFY(nearly_equal_within_tol(pr->c[j], cs[j], err_tol), "%luth column sum; got %e, expected %e",
                  j, cs[j], pr->c[j]);
    free(cs);
}

void verify_sols_equal(ds_problem *pr, ds_sol *sol1, ds_sol *sol2) {
    verify_vecs_equal(sol1->x, sol2->x, pr->m);
    verify_vecs_equal(sol1->y, sol2->y, pr->n);
    verify_mats_equal(sol1->p, sol2->p, pr->m, pr->n);
}


// Tests

void test_pseudo_intrinsics(void) {
    double x_arr[] = {1.23, -24.3, 0.4, 1e7};
    size_t n = sizeof(x_arr) / sizeof(double);

    __m256d x = _mm256_loadu_pd(x_arr);
    DS_VERIFY(ds_mm256_rmin_pd(x) == -24.3, "reducing min");
    DS_VERIFY(ds_mm256_rmax_pd(x) == 1e7, "reducing max");

    double sum = 0.;
    for (size_t i = 0; i < n; ++i) sum += x_arr[i];
    DS_VERIFY(ds_mm256_radd_pd(x) == sum, "reducing add");

    double abs_dest[n];
    __m256d x_abs = ds_mm256_abs_pd(x);
    _mm256_storeu_pd(abs_dest, x_abs);
    for (size_t i = 0; i < n; ++i) {
        DS_VERIFY(abs_dest[i] == fabs(x_arr[i]), "incorrect absolute value");
    }
    printf("[pass] pseudo intrinsics\n");
}

void test_roundings(void) {
    DS_VERIFY(DS_ROUND_DOWN_TO_VEC_LANE_COUNT(4) == 4, "4 stays the same");
    DS_VERIFY(DS_ROUND_DOWN_TO_VEC_LANE_COUNT(5) == 4, "5 rounds down");
    DS_VERIFY(DS_ROUND_DOWN_TO_VEC_LANE_COUNT(13) == 12, "13 rounds down");
    DS_VERIFY(DS_ROUND_DOWN_TO_VEC_LANE_COUNT(235347) == 235344, "large number");
    DS_VERIFY(DS_ROUND_UP_TO_VEC_LANE_COUNT(8) == 8, "8 stays the same");
    DS_VERIFY(DS_ROUND_UP_TO_VEC_LANE_COUNT(21) == 24, "21 rounds up");
    DS_VERIFY(DS_ROUND_UP_TO_VEC_LANE_COUNT(45614) == 45616, "large number");
    printf("[pass] roundings\n");
}

void test_mat_sums(void) {
    const size_t m = 16, n = 21;
    double *a = ds_mat_alloc(m, n);
    DS_VERIFY(a, "a alloc failed");
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            a[m * j + i] = (double) ((i + 1) * n + (j + 1));
        }
    }

    double *row_sums = ds_vec_alloc(m);
    DS_VERIFY(row_sums, "row_sums alloc failed");
    row_sums[0] = 1231.; // must be ignored
    ds_mat_row_sums(a, row_sums, m, n);
    for (size_t i = 0; i < m; ++i) {
        double row_sum = 0.;
        for (size_t j = 0; j < n; ++j) {
            row_sum += a[j * m + i];
        }
        DS_VERIFY(nearly_equal(row_sum, row_sums[i]), "incorrect row sum");
    }
    free(row_sums);

    double *col_sums = ds_vec_alloc(n);
    DS_VERIFY(col_sums, "col_sums alloc failed");
    col_sums[0] = 1231.; // must be ignored
    ds_mat_col_sums(a, col_sums, m, n);
    for (size_t j = 0; j < n; ++j) {
        double col_sum = 0.;
        for (size_t i = 0; i < m; ++i) {
            col_sum += a[j * m + i];
        }
        DS_VERIFY(nearly_equal(col_sum, col_sums[j]), "incorrect row sum");
    }

    free(col_sums); free(a);
    printf("[pass] row and column sums\n");
}

void test_scale_entries(void) {
    const size_t m = 53, n = 71;
    double *orig = ds_mat_alloc(m, n);
    DS_VERIFY(orig, "orig alloc failed");
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            orig[m * j + i] = (double) m * (double) n;

    double *a = ds_mat_alloc(m, n);
    DS_VERIFY(a, "a alloc failed");
    for (size_t k = 0; k < m * n; ++k) a[k] = orig[k];

    double *row_facts = ds_vec_alloc(m);
    DS_VERIFY(row_facts, "row_facts alloc failed");
    for (size_t i = 0; i < m; ++i) row_facts[i] = (double) i + 3.5;
    ds_mat_scale_rows(a, row_facts, m, n);
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            double expected = orig[j * m + i] * row_facts[i];
            DS_VERIFY((nearly_equal(a[j * m + i], expected)), "incorrect scaled entry");
        }
    }
    free(row_facts);

    for (size_t k = 0; k < m * n; ++k) a[k] = orig[k];
    double *col_facts = ds_vec_alloc(n);
    DS_VERIFY(col_facts, "col_facts alloc failed");
    for (size_t j = 0; j < n; ++j) col_facts[j] = (double) j + 12.2;
    ds_mat_scale_cols(a, col_facts, m, n);
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            double expected = orig[j * m + i] * col_facts[j];
            DS_VERIFY(nearly_equal(a[j * m + i], expected), "incorrect scaled entry");
        }
    }
    free(col_facts);
    free(a);
    free(orig);
    printf("[pass] scaling by rows and columns\n");
}

void test_mat_sub_set_diag(void) {
    const size_t m = 30, n = 20, p = 5;
    double *a = ds_mat_alloc(m, n),
           *v = ds_vec_alloc(p);
    DS_VERIFY(a, "a alloc failed");
    DS_VERIFY(v, "a alloc failed");
    for (size_t k = 0; k < m * n; ++k) a[k] = -1.;
    for (size_t i = 0; i < p; ++i) v[i] = (double) (i + 1);

    // Set A[2:2+p-1, 3:3+p-1] <- diag(v).
    size_t i_st = 1, j_st = 2;
    ds_mat_sub_set_diag(&a[m * j_st + i_st], v, m - p, p);

    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            if (i < i_st || j < j_st || i >= i_st + p || j >= j_st + p) {
                DS_VERIFY(a[m * j + i] == -1., "outside submatrix -1");
            } else if (i - i_st == j - j_st) {
                DS_VERIFY(a[m * j + i] == (double) (i - i_st + 1), "in main diagonal");
            } else {
                DS_VERIFY(!a[m * j + i], "outside diag but inside submatrix");
            }
        }
    }
    free(a); free(v);
    printf("[pass] mat_sub_set_diag\n");
}

void test_mat_prods(void) {
    const size_t m = 10, n = 5;
    double *a = ds_mat_alloc(m, n),
           *x = ds_vec_alloc(n),
           *y = ds_vec_alloc(m);
    DS_VERIFY(a, "a alloc failed");
    DS_VERIFY(x, "x alloc failed");
    DS_VERIFY(y, "y alloc failed");
    for (size_t k = 0; k < m * n; ++k) a[k] = (double) k;
    for (size_t j = 0; j < n; ++j) x[j] = (double) j;

    ds_mat_vec_prod(a, x, y, m, n);
    for (size_t i = 0; i < m; ++i) {
        double accum = 0.;
        for (size_t j = 0; j < n; ++j)
            accum = fma(a[m * j + i], x[j], accum);
        DS_VERIFY(nearly_equal(y[i], accum), "y[%lu] = %lf is incorrect, must be %lf",
                  i, y[i], accum);
    }

    for (size_t i = 0; i < m; ++i) y[i] = (double) i;
    ds_mat_t_vec_prod(a, y, x, m, n);
    for (size_t j = 0; j < n; ++j) {
        double accum = 0.;
        for (size_t i = 0; i < m; ++i)
            accum = fma(a[m * j + i], y[i], accum);
        DS_VERIFY(nearly_equal(x[j], accum), "x[%lu] = %lf must be %lf", j, x[j], accum);
    }

    free(a); free(x); free(y);
    printf("[pass] small matrix-vector products are correct\n");
}

void test_expl_and_impl(void) {
    const size_t m = 523, n = 123;
    const double err_tol = 1e-11;
    double *orig = ds_mat_alloc(m, n);
    DS_VERIFY(orig, "orig alloc failed");
    // Doesn't matter if a diagonal scaling of this matrix doesn't exist,
    // the results must also match if the limit of iterations is reached.
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            orig[m * j + i] = fabs(2. * (double) i - (double) j + 1.);

    ds_problem pr;
    ds_problem_init(&pr, m, n, 10, 1e-10);
    memcpy(pr.a, orig, m * n * sizeof(double));
    for (size_t i = 0; i < m; ++i) pr.r[i] = 1.0;
    for (size_t j = 0; j < n; ++j) pr.c[j] = (double) m / (double) n;

    ds_sol sol_expl1 = ds_expl_crit1(pr);
    verify_diag_scal_errs(&pr, &sol_expl1, err_tol);

    // The explicit solver may clobber the matrix A, restore.
    memcpy(pr.a, orig, m * n * sizeof(double));
    ds_sol sol_impl1 = ds_impl_crit1(pr);
    verify_sols_equal(&pr, &sol_expl1, &sol_impl1);
    ds_sol_free(&sol_expl1);
    ds_sol_free(&sol_impl1);

    memcpy(pr.a, orig, m * n * sizeof(double));
    ds_sol sol_expl2 = ds_expl_crit2(pr);
    verify_diag_scal_errs(&pr, &sol_expl2, err_tol);

    memcpy(pr.a, orig, m * n * sizeof(double));
    ds_sol sol_impl2 = ds_impl_crit2(pr);
    verify_sols_equal(&pr, &sol_expl2, &sol_impl2);
    ds_sol_free(&sol_expl2);
    ds_sol_free(&sol_impl2);

    memcpy(pr.a, orig, m * n * sizeof(double));
    ds_sol sol_expl3 = ds_expl_crit3(pr);
    verify_diag_scal_errs(&pr, &sol_expl3, err_tol);

    memcpy(pr.a, orig, m * n * sizeof(double));
    ds_sol sol_impl3 = ds_impl_crit3(pr);
    verify_sols_equal(&pr, &sol_expl3, &sol_impl3);
    ds_sol_free(&sol_expl3);
    ds_sol_free(&sol_impl3);

    ds_problem_free(&pr);
    free(orig);
    printf("[pass] expl and impl with stopping crits 1,2,3 give same solution\n");
}

void test_newt(void) {
    const size_t m = 10, n = 6;
    const double err_tol = 1e-11;
    double *orig = ds_mat_alloc(m, n);
    DS_VERIFY(orig, "orig alloc failed");
    // Doesn't matter if a diagonal scaling of this matrix doesn't exist,
    // the results must also match if the limit of iterations is reached.
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            orig[m * j + i] = (double) (i + j + 2);

    ds_problem pr;
    ds_problem_init(&pr, m, n, 5, 1e-4);
    memcpy(pr.a, orig, m * n * sizeof(double));
    for (size_t i = 0; i < m; ++i) pr.r[i] = 1.0;
    for (size_t j = 0; j < n; ++j) pr.c[j] = (double) m / (double) n;

    ds_sol sol_newt1 = ds_newt_crit1(pr);
    verify_diag_scal_errs(&pr, &sol_newt1, err_tol);
    ds_sol_free(&sol_newt1);

    /*mat_print(sol_newt1.p, pr.m, pr.n);
    mat_print(sol_newt1.x, 1, pr.m);
    mat_print(sol_newt1.y, 1, pr.n);
    printf("\niters = %u, res = %le\n", sol_newt1.iters, sol_newt1.res);*/

    // The solver may clobber the matrix A, restore.
    memcpy(pr.a, orig, m * n * sizeof(double));
    ds_sol sol_newt2 = ds_newt_crit2(pr);
    verify_diag_scal_errs(&pr, &sol_newt2, err_tol);
    ds_sol_free(&sol_newt2);

    memcpy(pr.a, orig, m * n * sizeof(double));
    ds_sol sol_newt3 = ds_newt_crit3(pr);
    verify_diag_scal_errs(&pr, &sol_newt3, err_tol);
    ds_sol_free(&sol_newt3);

    ds_problem_free(&pr); free(orig);
    printf("[pass] newt with stopping crits 1,2,3 runs successfully\n");
}

int main(void) {
    test_pseudo_intrinsics();
    test_roundings();
    test_mat_sums();
    test_scale_entries();
    test_mat_sub_set_diag();
    test_mat_prods();
    test_expl_and_impl();
    test_newt();
    ds_work_area_free();
}
