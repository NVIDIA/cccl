find_package(CUDAToolkit REQUIRED)

# Locate all test sources
file(
  GLOB_RECURSE test_srcs
  RELATIVE "${CUB_SOURCE_DIR}/test"
  CONFIGURE_DEPENDS
  catch2_test_*.cu
  test_*.cu
)

# Default Test Setup
add_library(cub.test.interface.default INTERFACE)
cccl_configure_target(cub.test.interface.default)
target_link_libraries(
  cub.test.interface.default
  INTERFACE #
    cub.compiler_interface
    CUDA::cuda_driver
)
target_compile_definitions(
  cub.test.interface.default
  INTERFACE #
    CCCL_ENABLE_ASSERTIONS
    CUB_DEBUG_SYNC
)
target_include_directories(
  cub.test.interface.default
  INTERFACE "${CUB_SOURCE_DIR}/test"
)

# Catch2 Setup
cccl_get_c2h()

function(_cub_is_catch2_src out_var test_src)
  set(${out_var} FALSE PARENT_SCOPE)
  if (test_src MATCHES "(^|/)catch2_test_")
    set(${out_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

# NVRTC test setup
if (CMAKE_CXX_COMPILER_ID STREQUAL NVHPC) # Unsupported
  list(FILTER test_srcs EXCLUDE REGEX "nvrtc")
else()
  configure_file(
    "cmake/nvrtc_args.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/nvrtc_args.h"
  )

  add_library(cub.test.interface.nvrtc INTERFACE)
  target_link_libraries(cub.test.interface.nvrtc INTERFACE CUDA::nvrtc)
  target_include_directories(
    cub.test.interface.nvrtc
    INTERFACE "${CMAKE_CURRENT_BINARY_DIR}"
  )
endif()

# NVTX test setup
# nvtx headers contain a variable named `module`, which breaks nvc++ as that is a keyword
if (CMAKE_CXX_COMPILER_ID STREQUAL NVHPC AND CMAKE_CUDA_STANDARD GREATER 17)
  list(FILTER test_srcs EXCLUDE REGEX "nvtx")
else()
  cccl_get_nvtx()
endif()

# Failure test setup
# Tests suffixed with `_fail` are expected to fail compilation and require
# customized setup.
function(_cub_is_fail_test out_var test_name)
  set(${out_var} FALSE PARENT_SCOPE)
  if (test_name MATCHES "\\.fail(\\.|$)")
    set(${out_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

# Launch Config Setup
#
# CUB's tests use the following convention for kernel launch methods:
#
# lid0 = host launch
# lid1 = device launch (requires CUB_ENABLE_RDC_TESTS)
# lid2 = graph launch
#
# To automatically build a test for different launch methods, adjust the following
# for the test's supported methods.
#
# // %PARAM% LAUNCHER lid 0:1:2
#
# All `device_*` tests that do not explicitly define %PARAM% comments
# will automatically build lid0 and lid1 variants. Fail tests (with `_fail.cu`
# suffix) are excluded from this behavior.
#
# Non-device tests and fail tests will never automatically add lid variants, but these can
# manually requested using %PARAM% comments.
#
# See these files for more details:
# - docs/cccl/development/testing.rst
# - cmake/ccclTestParams.cmake

option(
  CUB_ENABLE_LAUNCH_VARIANTS
  "Enable tests for device-side and graph-capture kernel launches"
  ON
)

set(cub_launcher_ids 0)
if (CUB_ENABLE_LAUNCH_VARIANTS)
  if (CUB_ENABLE_RDC_TESTS)
    list(APPEND cub_launcher_ids 1)
  endif()
  list(APPEND cub_launcher_ids 2)
endif()

function(_cub_set_rdc_with_launcher_id target_name lid)
  if (CUB_FORCE_RDC OR lid EQUAL 1)
    cccl_set_rdc(${target_name} TRUE)
  else()
    cccl_set_rdc(${target_name} FALSE)
  endif()
endfunction()

function(_cub_variant_label_to_launch_id lid_var label)
  set(${lid_var} "" PARENT_SCOPE) # No launcher requested.
  if (label MATCHES "(^|\\.)lid_([0-9]+)(\\.|$)")
    set(${lid_var} "${CMAKE_MATCH_2}" PARENT_SCOPE)
  endif()
endfunction()

# Convert known 'group_' prefixes to "group." for target naming.
function(_cub_test_prefix_to_group test_name_var)
  foreach (prefix IN_LISTS ARGN)
    if (prefix STREQUAL "${${test_name_var}}")
      set(${test_name_var} "${${test_name_var}}.basic")
    else()
      string(
        REGEX REPLACE
        "^${prefix}_"
        "${prefix}."
        ${test_name_var}
        "${${test_name_var}}"
      )
    endif()
  endforeach()
  set(${test_name_var} "${${test_name_var}}" PARENT_SCOPE)
endfunction()

# Special cases: Tweak build targets based on target name / file:
function(_cub_test_build_target_special_case build_target test_src)
  if (build_target MATCHES "^cub\\.test\\.nvtx_in_usercode.*")
    target_link_libraries(${build_target} PRIVATE nvtx3-cpp)
  endif()

  if (build_target MATCHES "\\.nvrtc")
    target_link_libraries(${build_target} PRIVATE cub.test.interface.nvrtc)
  endif()

  if (build_target MATCHES "^cub\\.test\\.iterator.*")
    # for handling large type lists:
    target_compile_options(${build_target} PRIVATE -ftemplate-depth=1000)
  endif()

  # enable lambdas for all API examples
  if (build_target MATCHES "^cub\\.test\\..+_api(\\.|$)")
    target_compile_options(
      ${build_target}
      PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
    )
  endif()

  # When clang >= 13 is used as host compiler, we get the following warning:
  #   nvcc_internal_extended_lambda_implementation:312:22: error: definition of implicit copy constructor for '__nv_hdl_wrapper_t<...>' is deprecated because it has a user-declared copy assignment operator [-Werror,-Wdeprecated-copy]
  #   312 | __nv_hdl_wrapper_t & operator=(const __nv_hdl_wrapper_t &in) = delete;
  #   |                      ^
  # Let's suppress it until NVBug 4980157 is resolved.
  # gersemi: off
  if (
    build_target STREQUAL "cub.test.device_segmented_scan_api" AND
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13
  )
    # gersemi: on
    target_compile_options(
      ${build_target}
      PRIVATE "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Wno-deprecated-copy>"
    )
  endif()
endfunction()

#
# Generic test assembler:
#

function(cub_add_test test_target test_src)
  set(options)
  set(oneValueArgs METATARGET_PATH)
  set(multiValueArgs)
  cmake_parse_arguments(
    self
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )
  cccl_parse_arguments_error_checks(
    "cub_add_test"
    self
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ERROR_UNPARSED
    DEFAULT_VALUES METATARGET_PATH "${test_target}"
  )

  cccl_add_executable(
    ${test_target}
    SOURCES "${test_src}"
    NO_METATARGETS # Added manually for non-fail tests
  )
  target_link_libraries(${test_target} PRIVATE cub.test.interface.default)

  _cub_is_catch2_src(is_catch2 "${test_src}")
  if (is_catch2)
    target_link_libraries(${test_target} PRIVATE cccl.c2h.main)
  else() # Non-catch2 tests leak catch2 headers from test_util.h, need c2h include paths:
    target_link_libraries(${test_target} PRIVATE cccl.c2h)
  endif()

  _cub_test_build_target_special_case(${test_target} "${test_src}")

  # Fail tests are handled specially:
  _cub_is_fail_test(is_fail "${test_target}")
  if (is_fail)
    cccl_add_xfail_compile_target_test(
      ${test_target}
      SOURCE_FILE "${test_src}"
      ERROR_REGEX_LABEL "expected-error"
      ERROR_NUMBER_TARGET_NAME_REGEX "\\.err_([0-9]+)"
    )
    return()
  endif()

  # Create metatargets for non-fail tests:
  cccl_ensure_metatargets(
    ${test_target}
    METATARGET_PATH ${self_METATARGET_PATH}
  )

  # Tell the `run_test.cmake` wrapper what kind of test this is:
  set(test_type "Basic")
  if (is_catch2)
    set(test_type "Catch2")
  endif()

  # Use the run_test.cmake wrapper; enables compute-sanitizer at runtime.
  add_test(
    NAME ${test_target}
    # gersemi: off
    COMMAND
      "${CMAKE_COMMAND}" "-DCCCL_SOURCE_DIR=${CCCL_SOURCE_DIR}"
      "-DTEST=$<TARGET_FILE:${test_target}>" "-DTYPE=${test_type}" -P
      "${CUB_SOURCE_DIR}/test/run_test.cmake"
    # gersemi: on
  )
  set_tests_properties(
    ${test_target}
    PROPERTIES SKIP_REGULAR_EXPRESSION "CCCL_SKIP_TEST"
  )
endfunction()

# Create tests from a source file, handling variants.
# All created tests are returned in test_targets_var.
function(cub_add_test_src test_targets_var test_src)
  # Prepare the test_name
  get_filename_component(test_name "${test_src}" NAME_WLE)
  string(REGEX REPLACE "^catch2_test_" "" test_name "${test_name}")
  string(REGEX REPLACE "^test_" "" test_name "${test_name}")
  # Format known group prefixes in test names:
  _cub_test_prefix_to_group(
    test_name
    thread
    warp
    block
    device
    util
  )
  string(PREPEND test_name "cub.test.")

  # Detect %PARAM% variants:
  cccl_parse_variant_params(
    "${test_src}"
    num_variants
    variant_labels
    variant_defs
  )

  if (num_variants EQUAL 0)
    cub_add_test(${test_name} "${test_src}")
    cccl_set_rdc(${test_name} ${CUB_FORCE_RDC})
    set(${test_targets_var} ${test_name} PARENT_SCOPE)
    return()
  endif()

  cccl_log_variant_params(
    ${test_name}
    ${num_variants}
    variant_labels
    variant_defs
  )

  # Subtract 1 to support the inclusive endpoint of foreach(...RANGE...):
  math(EXPR variant_range_end "${num_variants} - 1")
  set(test_targets)
  foreach (var_idx RANGE ${variant_range_end})
    cccl_get_variant_data(variant_labels variant_defs ${var_idx} label defs)

    # lid is an empty string if none requested in the variants:
    _cub_variant_label_to_launch_id(lid ${label})
    if (lid AND (NOT lid IN_LIST cub_launcher_ids))
      continue() # Disabled in current configuration
    endif()

    set(variant_name "${test_name}.${label}")
    cub_add_test(
      ${variant_name}
      "${test_src}"
      # Don't create metatargets for variable labels, just the base testname:
      METATARGET_PATH "${test_name}"
    )
    _cub_set_rdc_with_launcher_id(${variant_name} "${lid}")
    target_compile_definitions(${variant_name} PRIVATE ${defs})

    list(APPEND test_targets ${variant_name})
  endforeach() # Variant

  set(${test_targets_var} "${test_targets}" PARENT_SCOPE)
endfunction()

foreach (test_src IN LISTS test_srcs)
  cub_add_test_src(test_targets "${test_src}")
endforeach()

add_subdirectory(cmake)
add_subdirectory(ptx-json)
