find_package(CUDAToolkit REQUIRED)

#
# Locate all test sources
#

file(
  GLOB_RECURSE test_srcs
  RELATIVE "${CUB_SOURCE_DIR}/test"
  CONFIGURE_DEPENDS
  catch2_test_*.cu
  test_*.cu
)

#
# Default Test Setup
#

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
    CUB_DEBUG_SYNC
    CCCL_ENABLE_ASSERTIONS
)
target_include_directories(
  cub.test.interface.default
  INTERFACE "${CUB_SOURCE_DIR}/test"
)

#
# Catch2 Setup
#

cccl_get_c2h()
add_library(cub.test.interface.catch2 INTERFACE)
target_link_libraries(cub.test.interface.catch2 INTERFACE cccl.c2h.main)

function(_cub_is_catch2_src out_var test_src)
  set(${out_var} FALSE PARENT_SCOPE)
  if (test_src MATCHES "^catch2_test_" OR test_src MATCHES "/catch2_test_")
    set(${out_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

#
# NVRTC test setup
#

if (CMAKE_CXX_COMPILER_ID STREQUAL NVHPC) # Unsupported
  list(FILTER test_srcs EXCLUDE REGEX "nvrtc")
else()
  configure_file(
    "cmake/nvrtc_args.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/nvrtc_args.h"
  )

  add_library(cub.test.interface.nvrtc INTERFACE)
  target_include_directories(
    cub.test.interface.nvrtc
    INTERFACE "${CMAKE_CURRENT_BINARY_DIR}"
  )
endif()

#
# NVTX test setup
#

# nvtx headers contain a variable named `module`, which breaks nvc++ as that is a keyword
if (CMAKE_CXX_COMPILER_ID STREQUAL NVHPC AND CMAKE_CUDA_STANDARD GREATER 17)
  list(FILTER test_srcs EXCLUDE REGEX "nvtx")
else()
  cccl_get_nvtx()
  add_library(cub.test.interface.nvtx INTERFACE)
  target_link_libraries(cub.test.interface.nvtx INTERFACE nvtx3-cpp)
endif()

#
# Failure test setup
#

# Tests suffixed with `_fail` are expected to fail compilation and require
# customized setup.

function(_cub_is_fail_test out_var test_name)
  set(${out_var} FALSE PARENT_SCOPE)
  if (test_name MATCHES "\\.fail$" OR test_name MATCHES "\\.fail\\.")
    set(${out_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

#
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
  if (label MATCHES "\\.lid_([0-9]+)$" OR label MATCHES "\\.lid_([0-9]+)\\.")
    set(${lid_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
  endif()
endfunction()

#
# Test prefix groups: group similar tests using prefixes.
#

# Convert known 'group_' prefixes to "group." for target naming.
function(_cub_test_prefix_to_group test_name_var)
  string(REGEX REPLACE "^thread_" "thread." ${test_name_var} "${${test_name_var}}")
  string(REGEX REPLACE "^warp_" "warp." ${test_name_var} "${${test_name_var}}")
  string(REGEX REPLACE "^block_" "block." ${test_name_var} "${${test_name_var}}")
  string(REGEX REPLACE "^device_" "device." ${test_name_var} "${${test_name_var}}")
  string(REGEX REPLACE "^util_" "util." ${test_name_var} "${${test_name_var}}")
  set(${test_name_var} "${${test_name_var}}" PARENT_SCOPE)
endfunction()

# True if the test is in the `device` group.
function(_cub_is_device_test_name out_var test_name)
  set(${out_var} FALSE PARENT_SCOPE)
  if (test_name MATCHES "^device\\." OR test_name MATCHES "\\.device\\.")
    set(${out_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

#
# Special cases: Tweak build targets based on target name / file:
#

# Build targets:
function(_cub_test_build_target_special_case build_target test_src)
  if (build_target MATCHES "^cub\\.test\\.nvtx_in_usercode.*")
    target_link_libraries(${build_target} PRIVATE cub.test.interface.nvtx)
  endif()

  if (build_target MATCHES "\\.nvrtc")
    target_link_libraries(${build_target} PRIVATE cub.test.interface.nvrtc)
  endif()

  if (build_target MATCHES "^cub\\.test\\.iterator.*")
   # for handling large type lists:
   target_compile_options(${build_target} PRIVATE -ftemplate-depth=1000)
  endif()

  # enable lambdas for all API examples
  if (build_target MATCHES "^cub\\.test\\..+_api$")
    target_compile_options(
      ${build_target}
      PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
    )
  endif()

  # When clang >= 13 is used as host compiler, we get the following warning:
  #   nvcc_internal_extended_lambda_implementation:312:22: error: definition of implicit copy constructor for '__nv_hdl_wrapper_t<false, true, false, __nv_dl_tag<void (*)(), &TestAddressStabilityLambda, 2>, int (const int &)>' is deprecated because it has a user-declared copy assignment operator [-Werror,-Wdeprecated-copy]
  #   312 | __nv_hdl_wrapper_t & operator=(const __nv_hdl_wrapper_t &in) = delete;
  #   |                      ^
  # Let's suppress it until NVBug 4980157 is resolved.
  # gersemi: off
  if (build_target STREQUAL "cub.test.device.segmented_scan_api" AND
      CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
      # gersemi: on
    target_compile_options(
      ${build_target}
      PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: -Wno-deprecated-copy>
    )
  endif()
endfunction()

#
# Generic test assembler:
#

function(cub_add_test test_target test_src)
  cccl_parse_arguments(
    cub_add_test
    ONE_VALUE_ARGS METATARGET_PATH
    ERROR_IF_UNPARSED
    PARENT_ARGN ${ARGN}
  )
  if (NOT DEFINED self_METATARGET_PATH)
    set(self_METATARGET_PATH "$test_target")
  endif()

  cccl_add_executable(
    ${test_target}
    SOURCES "${test_src}"
    NO_METATARGETS # Added manually for non-fail tests
  )

  target_link_libraries(${test_target} PRIVATE cub.test.interface.default)

  _cub_is_catch2_src(is_catch2 "${test_src}")
  if (is_catch2)
    target_link_libraries(${test_target} PRIVATE cub.test.interface.catch2)
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
  cccl_ensure_metatargets(${test_target} METATARGET_PATH ${self_METATARGET_PATH})

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
      "${CMAKE_COMMAND}"
         "-DCCCL_SOURCE_DIR=${CCCL_SOURCE_DIR}"
         "-DTEST=$<TARGET_FILE:${test_target}>"
         "-DTYPE=${test_type}"
         -P "${CUB_SOURCE_DIR}/test/run_test.cmake"
    # gersemi: on
  )
  set_tests_properties(
    ${test_target}
    PROPERTIES SKIP_REGULAR_EXPRESSION "CCCL_SKIP_TEST"
  )
endfunction()

function(cub_add_test_src test_targets_var test_src)
  # Detect %PARAM% variants:
  cccl_parse_variant_params("${test_src}" num_variants variant_labels variant_defs)

  # Prepare the test_name
  get_filename_component(test_name "${test_src}" NAME_WE)
  if (DEFINED _cub_STRIP_PREFIX)
    string(REGEX REPLACE "^catch2_" "" test_name "${test_name}")
    string(REGEX REPLACE "^test_" "" test_name "${test_name}")
  endif()

  # Format known group prefixes in test names.
  # Needs to happen while there are no other prefixes:
  _cub_test_prefix_to_group(test_name)

  string(PREPEND test_name "cub.test.")

  # Automatically generate lid 0 / 1 for non-fail device tests that
  # aren't using variants.
  _cub_is_fail_test(is_fail "${test_name}")
  _cub_is_device_test(is_device "${test_name}")
  # gersemi: off
  if (num_variants EQUAL 0 AND is_device AND NOT is_fail)
    # gersemi: on
    set(num_variants 1)
    list(APPEND variant_labels "lid_0")
    list(APPEND variant_defs "TEST_LAUNCH=0")
    if (CUB_ENABLE_RDC_TESTS)
      set(num_variants 2)
      list(APPEND variant_labels "lid_1")
      list(APPEND variant_defs "TEST_LAUNCH=1")
    endif()
  endif()

  # If truly no variants, just add the standalone test:
  if (num_variants EQUAL 0)
    cub_add_test(${test_name} "${test_src}")
    cccl_set_rdc(${test_name} ${CUB_FORCE_RDC})
    target_link_libraries(${test_name} PRIVATE ${self_INTERFACE_LIBS})
    set(${test_targets_var} ${test_name} PARENT_SCOPE)
    return()
  endif()

  cccl_log_variant_params(
    "${test_name}"
    ${num_variants}
    variant_labels
    variant_defs
  )

  # Subtract 1 to support the inclusive endpoint of foreach(...RANGE...):
  math(EXPR variant_range_end "${num_variants} - 1")

  set(test_targets)
  foreach (var_idx RANGE ${variant_range_end})
    cccl_get_variant_data(variant_labels variant_defs ${var_idx} label defs)

    # Empty string if none requested:
    _cub_variant_label_to_launch_id(lid ${label})

    if (NOT lid IN_LIST cub_launcher_ids)
      continue() # Disabled in current configuration
    endif()

    set(variant_name "${test_name}.${label}")

    cub_add_test(
      ${variant_name}
      "${test_src}"
      # Don't create metatargets for variable labels, just the base testname:
      METATARGET_PATH "${test_name}"
    )
    _cub_set_rdc_with_launcher_id(${variant_name} ${lid})
    target_compile_definitions(${variant_name} PRIVATE ${defs})
    target_link_libraries(${variant_name} PRIVATE ${self_INTERFACE_LIBS})
    list(APPEND test_targets ${variant_name})
  endforeach() # Variant

  set(${test_targets_var} "${test_targets}" PARENT_SCOPE)
endfunction()

#
# Add Catch2 tests:
#

foreach (test_src IN_LISTS test_srcs)
  cub_add_test_src(test_targets "${test_src}")
endforeach()

add_subdirectory(cmake)
add_subdirectory(ptx-json)
