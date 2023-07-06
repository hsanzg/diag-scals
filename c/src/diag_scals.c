#include <diag_scals/diag_scals.h>
#include <diag_scals/common.h>

void ds_problem_init(ds_problem *pr, const size_t m, const size_t n,
                     const uint32_t max_iters, const double tol) {
    pr->a = ds_mat_alloc(m, n);
    pr->r = ds_vec_alloc(m); pr->c = ds_vec_alloc(n);
    pr->m = m; pr->n = n;
    pr->max_iters = max_iters;
    pr->tol = tol;
}

void ds_problem_free(ds_problem *pr) {
    free(pr->a);
    free(pr->r);
    free(pr->c);
}

ds_sol ds_sol_init(const ds_problem *pr) {
    size_t entry_count = DS_ROUND_UP_TO_VEC_LANE_COUNT(pr->m) + pr->n;
    double *diags = aligned_alloc(DS_VEC_BYTE_WIDTH, entry_count * sizeof(double));
    if (!diags) return (ds_sol){ .iters = DS_ERROR };
    ds_sol sol = {
            .x = diags,
            .y = &diags[DS_ROUND_UP_TO_VEC_LANE_COUNT(pr->m)],
            .iters = 1
    };
    // Set initial iterates to unity.
    for (size_t i = 0; i < pr->m; ++i) sol.x[i] = 1.;
    for (size_t j = 0; j < pr->n; ++j) sol.y[j] = 1.;
    return sol;
}

void ds_sol_free(ds_sol *sol) {
    free(sol->x);
    // y was in the same allocation as x.
    sol->x = sol->y = NULL;
}

bool ds_sol_is_err(const ds_sol *sol) {
    return sol->iters == DS_ERROR;
}

// Work areas.

/// Records metadata about a region of memory that can be used to
/// store auxiliary information during the computation of a
/// diagonal scaling. For example, the explicit solvers use the
/// underlying array to store the incremental factors by which
/// the diagonal entries of X or Y are multiplied at each iteration.
/// A work area must be freed prior to program exit by calling
/// `ds_work_area_free`; this includes the global work area.
typedef struct {
    size_t size;
    double *mem;
} ds_work_area;

/// A work area reserved for general use by any diagonal scaling
/// routine.
ds_work_area global_work_area;

double *ds_work_area_get(size_t cap) {
    if (!global_work_area.mem || global_work_area.size < cap) {
        // Not enough space, reallocate. There's unfortunately
        // no aligned variant for realloc, so free and realloc
        // manually. The contents of the previous array are not
        // preserved.
        if (global_work_area.mem) free(global_work_area.mem);
        global_work_area.mem = ds_vec_alloc(cap);
        global_work_area.size = global_work_area.mem ? cap : 0;
    }
    return global_work_area.mem;
}

void ds_work_area_free(void) {
    if (global_work_area.mem) free(global_work_area.mem);
    global_work_area.size = 0;
}
