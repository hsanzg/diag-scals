cmake_minimum_required(VERSION 3.25)
project(
  DiagonalScalings
  VERSION 1.0
  DESCRIPTION "A collection of diagonal scaling algorithms"
  LANGUAGES C CXX)

# Enable benchmarking.
set(BUILD_BENCHMARKING ON)
# Generate documentation with Doxygen.
set(BUILD_DOCS ON)

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

# The Doxygen documentation generator.
find_package(Doxygen)

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

# Enable documentation generation.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_DOCS AND DOXYGEN_FOUND)
  doxygen_add_docs(
    docs_diag_scals
    ${PROJECT_SOURCE_DIR}
    COMMENT "Generate API documentation")
endif()
