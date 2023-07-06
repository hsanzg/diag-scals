add_executable(bench_diag_scals bench.cpp)

target_link_libraries(bench_diag_scals PRIVATE benchmark::benchmark diag_scals)

target_compile_features(bench_diag_scals PRIVATE cxx_std_17)

target_compile_options(bench_diag_scals PRIVATE -Wall -Wextra -Wno-deprecated -Werror -march=native)

target_link_libraries(bench_diag_scals PRIVATE diag_scals)
