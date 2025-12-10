find_package(CUDAToolkit REQUIRED)

# Catch2 Setup
cccl_get_c2h()

function(_cub_is_catch2_src out_var test_src)
  set(${out_var} FALSE PARENT_SCOPE)
  if (test_src MATCHES "(^|/)catch2_test_")
    set(${out_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

# All tests link to this:
add_library(cub.test.interface.default INTERFACE)
cccl_configure_target(cub.test.interface.default)
target_link_libraries(
  cub.test.interface.default
  INTERFACE #
    cub.compiler_interface
    cccl.c2h # test_util.h uses c2h headers, even for non-catch2 tests.
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

# NVRTC test setup
if (CMAKE_CXX_COMPILER_ID STREQUAL NVHPC) # Unsupported
  set(nvrtc_supported FALSE)
else()
  set(nvrtc_supported TRUE)
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
  set(nvtx_supported FALSE)
else()
  set(nvtx_supported TRUE)
  cccl_get_nvtx()
endif()

# Convert known 'group_' prefixes to "group." for target naming.
#
set(
  _cub_test_group_prefixes
  thread
  warp
  block
  device
  util
)
function(_cub_test_prefix_to_group test_name_var)
  foreach (prefix IN_LISTS _cub_test_group_prefixes)
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

# Launch Config Setup
#
# CUB's tests use the following convention for kernel launch methods:
#
# lid0 = host launch
# lid1 = device launch (requires CUB_ENABLE_RDC_TESTS)
# lid2 = graph launch
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

# Extracts launcher ID from variant label.
# If no launcher is requested, the output variable is set to an empty string.
function(_cub_test_name_to_launch_id lid_var test_name)
  set(${lid_var} "" PARENT_SCOPE) # No launcher requested.
  if (test_name MATCHES "\\.lid_([0-9]+)(\\.|$)")
    set(${lid_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
  endif()
endfunction()

#
# Customization points:
#

# Set out_var to TRUE if the source file is supported, FALSE otherwise.
#
function(_cub_test_is_src_supported out_var test_src)
  set(result TRUE)
  if (test_src MATCHES "nvrtc" AND NOT nvrtc_supported)
    set(result FALSE)
  endif()

  if (test_src MATCHES "nvtx" AND NOT nvtx_supported)
    set(result FALSE)
  endif()

  set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

# Convert the test_src to a testcase basename.
# e.g. `catch2_test_device_foo.cu` -> `cub.test.device.foo`
#
function(_cub_test_src_to_basename out_var test_src)
  # Prepare the test_name
  get_filename_component(test_name "${test_src}" NAME_WLE)
  string(REGEX REPLACE "^catch2_test_" "" test_name "${test_name}")
  string(REGEX REPLACE "^test_" "" test_name "${test_name}")
  # Format known group prefixes in test names:
  _cub_test_prefix_to_group(test_name)
  string(PREPEND test_name "cub.test.")
  set(${out_var} "${test_name}" PARENT_SCOPE)
endfunction()

# Set out_var to TRUE if the testcase variant name is supported, FALSE otherwise.
#
function(_cub_test_is_test_supported out_var test_src test_name)
  set(result TRUE)
  # lid is an empty string if none requested in the variants:
  _cub_test_name_to_launch_id(lid ${test_name})
  if (lid AND (NOT lid IN_LIST cub_launcher_ids))
    set(result FALSE)
  endif()

  set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

# Link libraries, set defs, etc on the test executable target.
#
function(
  _cub_test_customize_executable
  test_src
  test_name
  test_executable_target
)
  _cub_test_name_to_launch_id(lid ${test_name})
  _cub_is_catch2_src(is_catch2 "${test_src}")

  target_link_libraries(
    ${test_executable_target}
    PRIVATE cub.test.interface.default
  )
  if (is_catch2)
    target_link_libraries(${test_executable_target} PRIVATE cccl.c2h.main)
  endif()

  if (test_name MATCHES "^cub\\.test\\.nvtx_in_usercode.*")
    target_link_libraries(${test_executable_target} PRIVATE nvtx3-cpp)
  endif()

  if (test_name MATCHES "\\.nvrtc")
    target_link_libraries(
      ${test_executable_target}
      PRIVATE cub.test.interface.nvrtc
    )
  endif()

  if (test_name MATCHES "^cub\\.test\\.iterator.*")
    # for handling large type lists:
    target_compile_options(
      ${test_executable_target}
      PRIVATE -ftemplate-depth=1000
    )
  endif()

  # enable lambdas for all API examples
  if (test_name MATCHES "^cub\\.test\\..+_api(\\.|$)")
    target_compile_options(
      ${test_executable_target}
      PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
    )
  endif()

  # NVBug 4980157: `__nv_hdl_wrapper_t` copy constructor deprecated warnings in Clang 13+
  # gersemi: off
  if (
    test_name STREQUAL "cub.test.device_segmented_scan_api" AND
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13
  )
    # gersemi: on
    target_compile_options(
      ${test_executable_target}
      PRIVATE "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Wno-deprecated-copy>"
    )
  endif()

  if (CUB_FORCE_RDC OR lid EQUAL 1)
    cccl_set_rdc(${test_executable_target} TRUE)
  else()
    cccl_set_rdc(${test_executable_target} FALSE)
  endif()
endfunction()

# Generate the command to run the test. Only for non-fail tests.
#
function(
  _cub_test_customize_test_command
  out_var
  test_src
  test_name
  test_executable_target
)
  _cub_is_catch2_src(is_catch2 "${test_src}")

  if (is_catch2)
    set(test_type "Catch2")
  else()
    set(test_type "Basic")
  endif()

  # gersemi: off
  # Run through run_test.cmake to enable compute-sanitizer at runtime.
  set(${out_var}
    "${CMAKE_COMMAND}"
      "-DCCCL_SOURCE_DIR=${CCCL_SOURCE_DIR}"
      "-DTEST=$<TARGET_FILE:${test_executable_target}>"
      "-DTYPE=${test_type}"
      -P "${CUB_SOURCE_DIR}/test/run_test.cmake"
    PARENT_SCOPE
  )
  # gersemi: on
endfunction()

# Set any additional properties on the CTest (including fail tests):
#
function(
  _cub_test_customize_test_properties
  test_src
  test_name
  test_executable_target
  ctest_name
)
  # Always skip tests that print CCCL_SKIP_TEST:
  set_tests_properties(
    ${ctest_name}
    PROPERTIES SKIP_REGULAR_EXPRESSION "CCCL_SKIP_TEST"
  )

  # xfail the tests that check for RDC based on launcher ID when forcing RDC:
  if (CUB_FORCE_RDC AND test_name MATCHES "cdp_variant_state")
    # Only the lid_0 and lid_2 variants will fail, lid_1 always uses RDC:
    if (NOT test_name MATCHES "\\.lid_1(\\.|$)")
      set_tests_properties(${ctest_name} PROPERTIES WILL_FAIL TRUE)
    endif()
  endif()
endfunction()

#
# Add tests:
#

file(
  GLOB_RECURSE test_srcs
  RELATIVE "${CUB_SOURCE_DIR}/test"
  CONFIGURE_DEPENDS
  catch2_test_*.cu
  test_*.cu
)

foreach (test_src IN LISTS test_srcs)
  cccl_add_tests_from_src(
    test_targets
    "${test_src}"
    FUNC_IS_SRC_SUPPORTED _cub_test_is_src_supported
    FUNC_SRC_TO_BASENAME _cub_test_src_to_basename
    FUNC_IS_TEST_SUPPORTED _cub_test_is_test_supported
    FUNC_CUSTOMIZE_EXECUTABLE _cub_test_customize_executable
    FUNC_CUSTOMIZE_TEST_COMMAND _cub_test_customize_test_command
    FUNC_CUSTOMIZE_TEST_PROPERTIES _cub_test_customize_test_properties
  )
endforeach()

add_subdirectory(cmake)
add_subdirectory(ptx-json)
