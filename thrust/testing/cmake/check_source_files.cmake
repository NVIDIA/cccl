# Check all source files for various issues that can be detected using pattern
# matching.
#
# This is run as a ctest test named `thrust.test.cmake.check_source_files`, or
# manually with:
# cmake -D "Thrust_SOURCE_DIR=<thrust project root>" -P check_source_files.cmake

cmake_minimum_required(VERSION 3.15)

function(count_substrings input search_regex output_var)
  string(REGEX MATCHALL "${search_regex}" matches "${input}")
  list(LENGTH matches num_matches)
  set(${output_var} ${num_matches} PARENT_SCOPE)
endfunction()

set(found_errors 0)
file(
  GLOB_RECURSE thrust_srcs
  RELATIVE "${Thrust_SOURCE_DIR}"
  "${Thrust_SOURCE_DIR}/thrust/*.h"
  "${Thrust_SOURCE_DIR}/thrust/*.inl"
)

################################################################################
# Namespace checks.
# Check all files in thrust to make sure that they use
# THRUST_NAMESPACE_BEGIN/END instead of bare `namespace thrust {}` declarations.
set(
  namespace_exclusions
  # This defines the macros and must have bare namespace declarations:
  thrust/detail/config/namespace.h
)

set(bare_ns_regex "namespace[ \n\r\t]+thrust[ \n\r\t]*\\{")

# Validation check for the above regex:
count_substrings([=[
namespace thrust{
namespace thrust {
namespace  thrust  {
 namespace thrust {
namespace thrust
{
namespace
thrust
{
]=]
  ${bare_ns_regex} valid_count
)
if (NOT valid_count EQUAL 6)
  message(
    FATAL_ERROR
    "Validation of bare namespace regex failed: "
    "Matched ${valid_count} times, expected 6."
  )
endif()

################################################################################
# stdpar header checks.
# Check all files in Thrust to make sure that they aren't including <algorithm>
# or <memory>, both of which will cause circular dependencies in nvc++'s
# stdpar library.
#
# The headers following headers should be used instead:
# <algorithm> -> <cuda/std/__host_stdlib/algorithm>
# <memory>    -> <cuda/std/__host_stdlib/memory>
# <numeric>   -> <cuda/std/__host_stdlib/numeric>
#
set(
  stdpar_header_exclusions
  # Placeholder -- none yet.
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
foreach (src ${thrust_srcs})
  file(READ "${Thrust_SOURCE_DIR}/${src}" src_contents)

  if (NOT ${src} IN_LIST namespace_exclusions)
    count_substrings("${src_contents}" "${bare_ns_regex}" bare_ns_count)
    count_substrings("${src_contents}" THRUST_NS_PREFIX prefix_count)
    count_substrings("${src_contents}" THRUST_NS_POSTFIX postfix_count)
    count_substrings("${src_contents}" THRUST_NAMESPACE_BEGIN begin_count)
    count_substrings("${src_contents}" THRUST_NAMESPACE_END end_count)
    count_substrings("${src_contents}" "#include <thrust/detail/config.h>" header_count
    )

    if (NOT bare_ns_count EQUAL 0)
      message(
        "'${src}' contains 'namespace thrust {...}'. Replace with THRUST_NAMESPACE macros."
      )
      set(found_errors 1)
    endif()

    if (NOT prefix_count EQUAL 0)
      message(
        "'${src}' contains 'THRUST_NS_PREFIX'. Replace with THRUST_NAMESPACE macros."
      )
      set(found_errors 1)
    endif()

    if (NOT postfix_count EQUAL 0)
      message(
        "'${src}' contains 'THRUST_NS_POSTFIX'. Replace with THRUST_NAMESPACE macros."
      )
      set(found_errors 1)
    endif()

    if (NOT begin_count EQUAL end_count)
      message("'${src}' namespace macros are unbalanced:")
      message(" - THRUST_NAMESPACE_BEGIN occurs ${begin_count} times.")
      message(" - THRUST_NAMESPACE_END   occurs ${end_count} times.")
      set(found_errors 1)
    endif()

    if (begin_count GREATER 0 AND header_count EQUAL 0)
      message(
        "'${src}' uses Thrust namespace macros, but does not (directly) `#include <thrust/detail/config.h>`."
      )
      set(found_errors 1)
    endif()
  endif()

  if (NOT ${src} IN_LIST stdpar_header_exclusions)
    count_substrings("${src_contents}" "${algorithm_regex}" algorithm_count)
    count_substrings("${src_contents}" "${memory_regex}" memory_count)
    count_substrings("${src_contents}" "${numeric_regex}" numeric_count)

    if (NOT algorithm_count EQUAL 0)
      message(
        "'${src}' includes the <algorithm> header. Replace with <cuda/std/__host_stdlib/algorithm>."
      )
      set(found_errors 1)
    endif()

    if (NOT memory_count EQUAL 0)
      message(
        "'${src}' includes the <memory> header. Replace with <cuda/std/__host_stdlib/memory>."
      )
      set(found_errors 1)
    endif()

    if (NOT numeric_count EQUAL 0)
      message(
        "'${src}' includes the <numeric> header. Replace with <cuda/std/__host_stdlib/numeric>."
      )
      set(found_errors 1)
    endif()
  endif()
endforeach()

################################################################################
# Deprecated-compiler warning ordering check in system headers.
# config.h issues a `_CCCL_WARNING` when the host compiler is below Thrust's
# minimum version, which expands to a `#pragma GCC/clang warning`. GCC and Clang
# silently drop diagnostics in files marked as system header.
# So the deprecated-compiler check must precede the system-header pragma in config.h,
# or the warning goes silent with no error. See NVIDIA/cccl#1620.
set(deprecated_compiler_marker "CCCL_IGNORE_DEPRECATED_COMPILER")
set(system_header_marker "_CCCL_IMPLICIT_SYSTEM_HEADER_GCC")
set(thrust_config_h "${Thrust_SOURCE_DIR}/thrust/detail/config/config.h")

file(READ "${thrust_config_h}" thrust_config_h_contents)
string(
  FIND "${thrust_config_h_contents}"
  "${deprecated_compiler_marker}"
  deprecated_compiler_pos
)
string(
  FIND "${thrust_config_h_contents}"
  "${system_header_marker}"
  system_header_pos
)

if (deprecated_compiler_pos EQUAL -1 OR system_header_pos EQUAL -1)
  message(
    "'${thrust_config_h}' no longer contains the expected ${deprecated_compiler_marker} (deprecated-compiler) "
    "or ${system_header_marker} (system-header) macros. Update this check if either moved or was renamed."
  )
  set(found_errors 1)
elseif (NOT deprecated_compiler_pos LESS system_header_pos)
  message(
    "'${thrust_config_h}' marks itself a system header before checking for a "
    "deprecated compiler. GCC/Clang silently drop diagnostics originating "
    "from system headers, so this ordering must be preserved (see NVIDIA/cccl#1620)."
  )
  set(found_errors 1)
endif()

if (NOT found_errors EQUAL 0)
  message(FATAL_ERROR "Errors detected.")
endif()
