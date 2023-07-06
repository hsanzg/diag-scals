#include <mkl_cblas.h>
#include <diag_scals/common.h>

double *ds_vec_alloc(const size_t n) {
    const size_t n_padded = DS_ROUND_UP_TO_VEC_LANE_COUNT(n);
    // caller is responsible for checking alloc errors.
    return aligned_alloc(DS_VEC_BYTE_WIDTH, n_padded * sizeof(double));
//    if (addr) {
//        for (size_t i = n; i < n_padded; ++i)
//            addr[i] = 0.;
//    }
}

double *ds_mat_alloc(const size_t m, const size_t n) {
    // caller is responsible for checking alloc errors.
    return malloc(m * n * sizeof(double));
}

void ds_mat_sub_set_diag(double *__restrict a, const double *__restrict v,
                         const size_t i_off, const size_t n) {
    v = __builtin_assume_aligned(v, DS_VEC_BYTE_WIDTH);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            a[(n + i_off) * j + i] = j == i ? v[i] : 0.;
}

void ds_mat_row_sums(const double *__restrict a, double *__restrict v,
                     const size_t m, const size_t n) {
    v = __builtin_assume_aligned(v, DS_VEC_BYTE_WIDTH);
    // todo: this zeroing process might be expensive, see how to avoid.
    memset(v, 0, m * sizeof(double));
    for (size_t j = 0; j < n; ++j) {
        // Iterate in blocks of VEC_LANE_COUNT lanes, because that's how many
        // doubles fit in an AVX2 vector register.
        size_t i = 0;
        for (; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(m); i += DS_VEC_LANE_COUNT) {
            __m256d accum = _mm256_load_pd(&v[i]);
            __m256d x = _mm256_loadu_pd(&a[m * j + i]);
            _mm256_store_pd(&v[i], _mm256_add_pd(accum, x));
        }
        // Handle remaining entries in jth column.
        for (; i < m; ++i) v[i] += a[m * j + i];
    }
}

void ds_mat_col_sums(const double *__restrict a, double *__restrict v,
                     const size_t m, const size_t n) {
    v = __builtin_assume_aligned(v, DS_VEC_BYTE_WIDTH);
    for (size_t j = 0; j < n; ++j) {
        // Perform reduction over the jth column of A.
        size_t i = 0;
        // Unroll two iterations to saturate the load ports.
        __m256d accum_1 = _mm256_setzero_pd(),
                accum_2 = _mm256_setzero_pd();
        for (; i < DS_ROUND_DOWN(m, 2 * DS_VEC_LANE_COUNT); i += 2 * DS_VEC_LANE_COUNT) {
            __m256d x_1 = _mm256_loadu_pd(&a[m * j + i]);
            __m256d x_2 = _mm256_loadu_pd(&a[m * j + i + DS_VEC_LANE_COUNT]);
            accum_1 = _mm256_add_pd(accum_1, x_1);
            accum_2 = _mm256_add_pd(accum_2, x_2);
        }
        v[j] = ds_mm256_radd_pd(_mm256_add_pd(accum_1, accum_2));
        for(; i < m; ++i) v[j] += a[m * j + i];
    }
}

void ds_mat_scale_rows(double *__restrict a, const double *__restrict v,
                       const size_t m, const size_t n) {
    v = __builtin_assume_aligned(v, DS_VEC_BYTE_WIDTH);
    for (size_t j = 0; j < n; ++j) {
        size_t i = 0;
        for (; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(m); i += DS_VEC_LANE_COUNT) {
            __m256d x = _mm256_loadu_pd(&a[m * j + i]);
            __m256d factors = _mm256_load_pd(&v[i]);
            _mm256_storeu_pd(&a[m * j + i], _mm256_mul_pd(x, factors));
        }
        for (; i < m; ++i) a[m * j + i] *= v[i];
    }
}

void ds_mat_scale_rows_with_inc(const double *__restrict a, const double *__restrict v, double *__restrict b,
                                const size_t i_off, const size_t m, const size_t n) {
    v = __builtin_assume_aligned(v, DS_VEC_BYTE_WIDTH);
    for (size_t j = 0; j < n; ++j) {
        size_t i = 0;
        for (; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(m); i += DS_VEC_LANE_COUNT) {
            __m256d x = _mm256_loadu_pd(&a[m * j + i]);
            __m256d factors = _mm256_load_pd(&v[i]);
            _mm256_storeu_pd(&b[(m + i_off) * j + i], _mm256_mul_pd(x, factors));
        }
        for (; i < m; ++i) b[(m + i_off) * j + i] = v[i] * a[m * j + i];
    }
}

void ds_mat_t_scale_rows_with_inc(const double *__restrict a, const double *__restrict v, double *__restrict b,
                                  const size_t i_off, const size_t m, const size_t n) {
    v = __builtin_assume_aligned(v, DS_VEC_BYTE_WIDTH);
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            // AVX2 doesn't support scatters, do writes sequentially.
            // This is quite inefficient, we traverse B by rows.
            b[(n + i_off) * i + j] = v[j] * a[m * j + i];
        }
    }
}

void ds_mat_scale_cols(double *__restrict a, const double *__restrict v,
                       const size_t m, const size_t n) {
    v = __builtin_assume_aligned(v, DS_VEC_BYTE_WIDTH);
    for (size_t j = 0; j < n; ++j) {
        size_t i = 0;
        __m256d factors = _mm256_set1_pd(v[j]);
        for (; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(m); i += DS_VEC_LANE_COUNT) {
            __m256d x = _mm256_loadu_pd(&a[m * j + i]);
            _mm256_storeu_pd(&a[m * j + i], _mm256_mul_pd(x, factors));
        }
        for (; i < m; ++i) a[m * j + i] *= v[j];
    }
}

void ds_mat_scale_rows_cols(double *__restrict a, const double *__restrict x, const double *__restrict y,
                            const size_t m, const size_t n) {
    for (size_t j = 0; j < n; ++j) {
        size_t i = 0;
        __m256d ys = _mm256_set1_pd(y[j]);
        for (; i < DS_ROUND_DOWN_TO_VEC_LANE_COUNT(m); i += DS_VEC_LANE_COUNT) {
            __m256d xs = _mm256_load_pd(&x[i]);
            __m256d as = _mm256_loadu_pd(&a[m * j + i]);
            as = _mm256_mul_pd(xs, _mm256_mul_pd(as, ys));
            _mm256_storeu_pd(&a[m * j + i], as);
        }
        for (; i < m; ++i) {
            a[m * j + i] *= x[i] * y[j];
        }
    }
}

void ds_mat_vec_prod(const double *__restrict a, const double *__restrict x, double *__restrict y,
                  const size_t m, const size_t n) {
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, 1., a, m, x, 1, 0., y, 1);
}

void ds_mat_t_vec_prod(const double *__restrict a, const double *__restrict x, double *__restrict y,
                    const size_t m, const size_t n) {
    cblas_dgemv(CblasColMajor, CblasTrans, m, n, 1., a, m, x, 1, 0., y, 1);
}
