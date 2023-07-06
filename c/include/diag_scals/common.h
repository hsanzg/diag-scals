#ifndef DIAGONALSCALINGS_COMMON_H
#define DIAGONALSCALINGS_COMMON_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h> // size_t
#include <stdint.h> // uint32_t
#include <x86intrin.h> // __m*, _mm*
#include <string.h> // memcpy
#include <math.h> // fabs
#include <stdbool.h>


// Utility macros.

#define DS_FORCE_INLINE __attribute__((always_inline)) inline

#define DS_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define DS_MIN(a, b) (((a) < (b)) ? (a) : (b))


// AVX2 vector utilities.

/// The byte width of an AVX2 vector register.
#define DS_VEC_BYTE_WIDTH 32
/// The number of lanes of an AVX2 double-precision floating point vector register.
#define DS_VEC_LANE_COUNT (DS_VEC_BYTE_WIDTH / 8)

// `mult` must be a power of 2.
#define DS_ROUND_DOWN(n, mult) ((n) & ~(mult - 1))
#define DS_ROUND_DOWN_TO_VEC_LANE_COUNT(n) DS_ROUND_DOWN(n, DS_VEC_LANE_COUNT)
#define DS_ROUND_UP_TO_VEC_LANE_COUNT(n) (((n) + 3) & ~3)

static DS_FORCE_INLINE double ds_mm256_radd_pd(__m256d x) {
    __m128d x_lo = _mm256_castpd256_pd128(x);
    __m128d x_up = _mm256_extractf128_pd(x, 1);
    x_lo = _mm_add_pd(x_lo, x_up);
    __m128d up64 = _mm_unpackhi_pd(x_lo, x_lo);
    return _mm_cvtsd_f64(_mm_add_sd(x_lo, up64));
}

static DS_FORCE_INLINE double ds_mm256_rmin_pd(__m256d x) {
    __m128d x_lo = _mm256_castpd256_pd128(x);
    __m128d x_up = _mm256_extractf128_pd(x, 1);
    x_lo = _mm_min_pd(x_lo, x_up);
    __m128d up64 = _mm_unpackhi_pd(x_lo, x_lo);
    return _mm_cvtsd_f64(_mm_min_sd(x_lo, up64));
}

static DS_FORCE_INLINE double ds_mm256_rmax_pd(__m256d x) {
    __m128d x_lo = _mm256_castpd256_pd128(x);
    __m128d x_up = _mm256_extractf128_pd(x, 1);
    x_lo = _mm_max_pd(x_lo, x_up);
    __m128d up64 = _mm_unpackhi_pd(x_lo, x_lo);
    return _mm_cvtsd_f64(_mm_max_sd(x_lo, up64));
}

static DS_FORCE_INLINE __m256d ds_mm256_abs_pd(__m256d x) {
    __m256d sign_mask = _mm256_set1_pd(-0.0); // 1 << 63
    return _mm256_andnot_pd(sign_mask, x);
}


// Basic matrix and vector calculations.

/// Allocates memory for an nx1 vector, padded to the nearest `DS_VEC_BYTE_WIDTH`-byte
/// boundary with zero entries. The returned address is a multiple of `DS_VEC_BYTE_WIDTH`
/// bytes, so that the vector can benefit from aligned AVX2 loads and stores.
double *ds_vec_alloc(size_t n);

/// Allocates memory for an mxn matrix stored in column-major order.
double *ds_mat_alloc(size_t m, size_t n);

/// Copies the entries of the nx1 vector v to the main diagonal of the nxn submatrix
/// of A stored in column-major order, starting at row `i_off`. Also clears
/// the entries outside of the main diagonal of the submatrix.
///
/// # Preconditions
/// The address of v is a multiple of `DS_VEC_BYTE_WIDTH` bytes.
void ds_mat_sub_set_diag(double *a, const double *v, size_t i_off, size_t n);

/// Computes the row sums of the mxn matrix A stored in column-major order,
/// placing the results in the mx1 vector v.
///
/// # Preconditions
/// - The address of v is a multiple of `DS_VEC_BYTE_WIDTH` bytes.
void ds_mat_row_sums(const double *a, double *v, size_t m, size_t n);

/// Computes the column sums of the mxn matrix A stored in column-major order,
/// placing the results in the nx1 vector v.
///
/// # Preconditions
/// The address of v is a multiple of `DS_VEC_BYTE_WIDTH` bytes.
void ds_mat_col_sums(const double *a, double *v, size_t m, size_t n);

/// Scales the rows of the mxn matrix A stored in column-major order by
/// the corresponding entries of the mx1 vector v.
///
/// # Preconditions
/// The address of v is a multiple of `DS_VEC_BYTE_WIDTH` bytes.
void ds_mat_scale_rows(double *a, const double *v, size_t m, size_t n);

/// Scales the rows of the mxn matrix A stored in column-major order by
/// the corresponding entries of the mx1 vector v. The results are placed on
/// the mxn matrix B in "hopping" column-major order. That is, the jth column
/// of B starts at address `&b[(m + i_off) * (j - 1)]`.  For example, to place
/// the 2x2 resulting matrix in the 2x2 block of B starting at (3, 1), pass
/// the address of b_{3,1} and set `i_off = 3 - 1`.
///
/// # Preconditions
/// The address of v is a multiple of `DS_VEC_BYTE_WIDTH` bytes.
void ds_mat_scale_rows_with_inc(const double *a, const double *v, double *b,
                                size_t i_off, size_t m, size_t n);

/// Scales the rows of A' by the corresponding entries of the nx1 vector x,
/// where A is an mxn matrix stored in column-major order. The results
/// are placed on the nxm matrix B in "hopping" column-major order. That is,
/// the jth column of B starts at address `&b[(n + i_off) * (j - 1)]`.
/// This method should not be confused with `mat_scale_cols`, which computes
/// the transpose of this method: this function finds diag(x)*A'; the other
/// does A*diag(x).
///
/// # Preconditions
/// The address of v is a multiple of `DS_VEC_BYTE_WIDTH` bytes.
void ds_mat_t_scale_rows_with_inc(const double *a, const double *v, double *b,
                                  size_t i_off, size_t m, size_t n);

/// Scales the columns of the mxn matrix A stored in column-major order by
/// the corresponding entries of the nx1 vector v.
///
/// # Preconditions
/// The address of v is a multiple of `DS_VEC_BYTE_WIDTH` bytes.
void ds_mat_scale_cols(double *a, const double *v, size_t m, size_t n);

/// Scales the rows of the mxn matrix A stored in column-major order by
/// the corresponding entries of the mx1 vector x; then scales the columns of
/// the result by the corresponding entries of the nx1 vector y.
///
/// # Preconditions
/// The addresses of x and y are multiples of `DS_VEC_BYTE_WIDTH` bytes.
void ds_mat_scale_rows_cols(double *a, const double *x, const double *y,
                            size_t m, size_t n);

/// Computes the product A*x, where A is an mxn matrix stored in column-major
/// order and x is an nx1 vector. The results are placed on the mx1 vector y.
///
/// # Preconditions
/// The addresses of x and y are multiples of `DS_VEC_BYTE_WIDTH` bytes.
void ds_mat_vec_prod(const double *a, const double *x, double *y,
                     size_t m, size_t n);

/// Computes the product A'*x, where A is an mxn matrix stored in
/// column-major order and x is an mx1 vector. The results are
/// placed on the nx1 vector y.
///
/// # Preconditions
/// The addresses of x and y are multiples of `DS_VEC_BYTE_WIDTH` bytes.
void ds_mat_t_vec_prod(const double *a, const double *x, double *y,
                       size_t m, size_t n);

/// Marker value used to indicate that an error occurred while executing
/// a diagonal scaling method; see `ds_sol_is_err`.
#define DS_ERROR UINT32_MAX

/// Marks and returns the solution as erroneous if the first argument is false.
#define DS_SOL_VERIFY(cond, sol) if (__builtin_expect(!(cond), 1)) { sol.iters = DS_ERROR; return sol; }

#ifdef __cplusplus
}
#endif
#endif //DIAGONALSCALINGS_COMMON_H
