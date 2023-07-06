add_executable(test_diag_scals test.c)

target_compile_features(test_diag_scals PRIVATE c_std_11)

target_compile_options(test_diag_scals PRIVATE -Wall -Wextra -Wno-deprecated -Werror -march=native)

target_link_libraries(test_diag_scals PRIVATE diag_scals)

add_test(NAME test_diag_scals_test COMMAND test_diag_scals)
