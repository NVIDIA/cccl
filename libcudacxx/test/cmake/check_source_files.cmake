# Check libcudacxx source files for issues that can be detected using pattern
# matching.
#
# This is run as a ctest test named `libcudacxx.test.cmake.check_source_files`,
# or manually with:
# cmake -D "LIBCUDACXX_SOURCE_DIR=<libcudacxx project root>" -P check_source_files.cmake

cmake_minimum_required(VERSION 3.15)

function(count_substrings input search_regex output_var)
  string(REGEX MATCHALL "${search_regex}" matches "${input}")
  list(LENGTH matches num_matches)
  set(${output_var} ${num_matches} PARENT_SCOPE)
endfunction()

set(found_errors 0)
file(
  GLOB_RECURSE libcudacxx_srcs
  RELATIVE "${LIBCUDACXX_SOURCE_DIR}"
  "${LIBCUDACXX_SOURCE_DIR}/include/cuda/*"
  "${LIBCUDACXX_SOURCE_DIR}/include/nv/*"
)

# Exclude the imported libc++ headers from the scan. They are not under CCCL's
# direct control and intentionally mirror the upstream libc++ implementation.
list(FILTER libcudacxx_srcs EXCLUDE REGEX "^include/cuda/std/detail/libcxx/")

################################################################################
# stdpar header checks.
# Check all files in libcudacxx to make sure that they aren't including
# <algorithm>, <memory>, or <numeric>, all of which can introduce circular
# dependencies with compilers that integrate CCCL components deeply into
# their C++ standard library implementations.
#
# The following headers should be used instead:
# <algorithm> -> <cuda/std/__cccl/algorithm_wrapper.h>
# <memory>    -> <cuda/std/__cccl/memory_wrapper.h>
# <numeric>   -> <cuda/std/__cccl/numeric_wrapper.h>
#
set(
  stdpar_header_exclusions
  include/cuda/std/__cccl/algorithm_wrapper.h
  include/cuda/std/__cccl/memory_wrapper.h
  include/cuda/std/__cccl/numeric_wrapper.h
)

set(algorithm_regex "#[ \t]*include[ \t]+<algorithm>")
set(memory_regex "#[ \t]*include[ \t]+<memory>")
set(numeric_regex "#[ \t]*include[ \t]+<numeric>")

# Validation check for the above regex pattern:
count_substrings([=[
#include <algorithm>
# include <algorithm>
#include  <algorithm>
# include  <algorithm>
# include  <algorithm> // ...
]=]
  ${algorithm_regex} valid_count
)
if (NOT valid_count EQUAL 5)
  message(
    FATAL_ERROR
    "Validation of stdpar header regex failed: "
    "Matched ${valid_count} times, expected 5."
  )
endif()

################################################################################
# Read source files:
foreach (src ${libcudacxx_srcs})
  if (IS_DIRECTORY "${LIBCUDACXX_SOURCE_DIR}/${src}")
    continue()
  endif()

  file(READ "${LIBCUDACXX_SOURCE_DIR}/${src}" src_contents)

  if (NOT ${src} IN_LIST stdpar_header_exclusions)
    count_substrings("${src_contents}" "${algorithm_regex}" algorithm_count)
    count_substrings("${src_contents}" "${memory_regex}" memory_count)
    count_substrings("${src_contents}" "${numeric_regex}" numeric_count)

    if (NOT algorithm_count EQUAL 0)
      message(
        "'${src}' includes the <algorithm> header. Replace with <cuda/std/__cccl/algorithm_wrapper.h>."
      )
      set(found_errors 1)
    endif()

    if (NOT memory_count EQUAL 0)
      message(
        "'${src}' includes the <memory> header. Replace with <cuda/std/__cccl/memory_wrapper.h>."
      )
      set(found_errors 1)
    endif()

    if (NOT numeric_count EQUAL 0)
      message(
        "'${src}' includes the <numeric> header. Replace with <cuda/std/__cccl/numeric_wrapper.h>."
      )
      set(found_errors 1)
    endif()
  endif()
endforeach()

if (NOT found_errors EQUAL 0)
  message(FATAL_ERROR "Errors detected.")
endif()
