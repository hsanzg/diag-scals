cmake_minimum_required(VERSION 3.25)
project(
  DiagonalScalings
  VERSION 1.0
  DESCRIPTION "A collection of diagonal scaling algorithms"
  LANGUAGES C CXX)

# Enable benchmarking.
set(BUILD_BENCHMARKING ON)

# Enable tests if this is the main project.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # Support directories in IDEs.
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  include(CTest)
endif()

# Provides BLAS and LAPACK implementations.
find_package(MKL CONFIG REQUIRED)

# The Google Benchmark library.
find_package(benchmark REQUIRED)

add_subdirectory(src)

# Enable testing if this is the main project.
# We provide the DIAG_SCALS_CMAKE_BUILD_TESTING override.
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR DIAG_SCALS_CMAKE_BUILD_TESTING)
        AND BUILD_TESTING)
  add_subdirectory(test)
endif()

# Enable benchmarking if this is the main project.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_BENCHMARKING)
  add_subdirectory(bench)
endif()
