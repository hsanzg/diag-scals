file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${DiagonalScalings_SOURCE_DIR}/include/diag_scals/*.h")

add_library(diag_scals common.c diag_scals.c ${HEADER_LIST} explicit.c implicit.c newton.c)

target_include_directories(diag_scals PUBLIC ../include)

target_link_libraries(diag_scals PRIVATE $<LINK_ONLY:MKL::MKL> m)
target_compile_options(diag_scals PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(diag_scals PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)

# Users of the library need at least C11.
target_compile_features(diag_scals PUBLIC c_std_11)

# Enable additional warnings, and make them all errors.
target_compile_options(diag_scals PRIVATE -Wall -Wextra -Wno-deprecated -Werror)

# Enable the AVX2 x86 extension.
target_compile_options(diag_scals PUBLIC -mavx2)

# Tell IDEs where to place header files.
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})
