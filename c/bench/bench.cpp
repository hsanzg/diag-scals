#include <benchmark/benchmark.h>
//#include <mkl.h>
//#include <omp.h>
#include <diag_scals/diag_scals.h>


// Benchmark variants

/// Create a mxn matrix stored in column-major order having
/// i*n+j as its (i, j) entry.
void fill_test_matrix(ds_problem *pr) {
    for (size_t j = 0; j < pr->n; ++j)
        for (size_t i = 0; i < pr->m; ++i)
            pr->a[pr->m * j + i] = (double) ((i + 1) * pr->n + (j + 1));
}

void setup_positive_mat_same_sums(ds_problem *pr) {
    fill_test_matrix(pr);
    for (size_t i = 0; i < pr->m; ++i)
        pr->r[i] = 1.0;
    for (size_t j = 0; j < pr->n; ++j)
        pr->c[j] = (double) pr->m / (double) pr->n; // so that sum(r) = sum(c)
}


// Benchmark setup code

template <class ...Args>
static void BM_diag_scal(benchmark::State& state, Args&&... args) {
    auto args_tuple = std::make_tuple(std::move(args)...);

    ds_problem pr;
    ds_problem_init(&pr, 1000, 1000, 10, 1e-20);
    for (auto _ : state) {
        state.PauseTiming();
        // Setup problem: generate matrix A and vectors r, c.
        (std::get<0>(args_tuple))(&pr);
        state.ResumeTiming();

        // Solve the problem using the specified solver routine.
        ds_sol sol = (std::get<1>(args_tuple))(pr);
        state.PauseTiming();
        benchmark::DoNotOptimize(sol);
        ds_sol_free(&sol);
    }
    ds_problem_free(&pr);
}


// Benchmarks

BENCHMARK_CAPTURE(BM_diag_scal, expl_crit1, setup_positive_mat_same_sums, ds_expl_crit1);
BENCHMARK_CAPTURE(BM_diag_scal, impl_crit1, setup_positive_mat_same_sums, ds_impl_crit1);
BENCHMARK_CAPTURE(BM_diag_scal, newt_crit1, setup_positive_mat_same_sums, ds_newt_crit1);

BENCHMARK_CAPTURE(BM_diag_scal, explicit_crit2, setup_positive_mat_same_sums, ds_expl_crit2);
BENCHMARK_CAPTURE(BM_diag_scal, impl_crit2, setup_positive_mat_same_sums, ds_impl_crit2);

BENCHMARK_CAPTURE(BM_diag_scal, explicit_crit3, setup_positive_mat_same_sums, ds_expl_crit3);
BENCHMARK_CAPTURE(BM_diag_scal, impl_crit3, setup_positive_mat_same_sums, ds_impl_crit3);

int main(int argc, char **argv) {
//    mkl_set_num_threads(8);
//    mkl_set_dynamic(0);
//    omp_set_num_threads(8);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    ds_work_area_free();
    return 0;
}
