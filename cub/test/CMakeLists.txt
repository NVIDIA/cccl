option(
  CUB_ENABLE_LAUNCH_VARIANTS
  "Enable CUB launch variants (lid_1 and lid_2)"
  ON
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if (CUB_ENABLE_RDC_TESTS)
    if ("${CMAKE_VERSION}" VERSION_LESS 3.27.5)
      # https://gitlab.kitware.com/cmake/cmake/-/merge_requests/8794
      message(
        WARNING
        "CMake 3.27.5 or newer is required to enable RDC tests in Visual Studio."
      )
      cmake_minimum_required(VERSION 3.27.5)
    endif()
  endif()
endif()

cccl_get_c2h()
cccl_get_nvtx()

find_package(CUDAToolkit)

set(build_nvrtc_tests ON)
if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  set(build_nvrtc_tests OFF)
endif()

file(
  GLOB_RECURSE test_srcs
  RELATIVE "${CUB_SOURCE_DIR}/test"
  CONFIGURE_DEPENDS
  test_*.cu
  catch2_test_*.cu
)

# nvtx headers contain a variable named `module`, which breaks nvc++ as that is a keyword
if (
  "NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}"
  AND NOT "${CMAKE_CXX_STANDARD}" MATCHES "17"
)
  list(FILTER test_srcs EXCLUDE REGEX "test_nvtx*")
endif()

## _cub_is_catch2_test
#
# If the test_src contains the substring "catch2_test_", `result_var` will
# be set to TRUE.
function(_cub_is_catch2_test result_var test_src)
  string(FIND "${test_src}" "catch2_test_" idx)
  if (idx EQUAL -1)
    set(${result_var} FALSE PARENT_SCOPE)
  else()
    set(${result_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

## _cub_is_fail_test
#
# If the test_src contains the substring "_fail", `result_var` will
# be set to TRUE.
function(_cub_is_fail_test result_var test_src)
  string(FIND "${test_src}" "_fail" idx)
  if (idx EQUAL -1)
    set(${result_var} FALSE PARENT_SCOPE)
  else()
    set(${result_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

## _cub_launcher_requires_rdc
#
# If given launcher id corresponds to a CDP launcher, set `out_var` to 1.
function(_cub_launcher_requires_rdc out_var launcher_id)
  if ("${launcher_id}" STREQUAL "1")
    set(${out_var} 1 PARENT_SCOPE)
  else()
    set(${out_var} 0 PARENT_SCOPE)
  endif()
endfunction()

## cub_add_test
#
# Add a test executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the test
#   target. Useful for modifying the test/target after creation.
# variant_suffix: Suffix to append to the test name to indicate variant
# test_name: The name of the test minus "<config_prefix>.test." For example,
#   testing/vector.cu will be "vector", and testing/cuda/copy.cu will be
#   "cuda.copy".
# test_src: The source file that implements the test.
# launcher_id: The launcher variant id for this test (0=host, 1=device, 2=graph)
#
function(
  cub_add_test
  target_name_var
  test_name
  variant_suffix
  test_src
  launcher_id
)
  _cub_is_catch2_test(is_catch2_test "${test_src}")
  _cub_launcher_requires_rdc(cdp_val "${launcher_id}")

  # The actual name of the test's target:
  set(test_target cub.test.${test_name}${variant_suffix})
  set(${target_name_var} ${test_target} PARENT_SCOPE)

  # The metatarget path ignores the variant suffix:
  set(metatarget_path cub.test.${test_name})

  if (is_catch2_test)
    # Per config helper library:
    set(config_c2h_target cub.test.catch2_helper.lid_${launcher_id})
    if (NOT TARGET ${config_c2h_target})
      add_library(${config_c2h_target} INTERFACE)
      cccl_configure_target(${config_c2h_target})
      cccl_ensure_metatargets(${config_c2h_target})
      cub_configure_cuda_target(${config_c2h_target} RDC ${cdp_val})
      target_include_directories(
        ${config_c2h_target}
        INTERFACE "${CUB_SOURCE_DIR}/test"
      )
      target_link_libraries(
        ${config_c2h_target}
        INTERFACE #
          cub.compiler_interface
          cccl.c2h
          CUDA::nvrtc
          CUDA::cuda_driver
      )
    endif() # config_c2h_target

    cccl_add_executable(
      ${test_target}
      SOURCES "${test_src}"
      METATARGET_PATH ${metatarget_path}
    )
    target_link_libraries(
      ${test_target}
      PRIVATE #
        cub.compiler_interface
        ${config_c2h_target}
        cccl.c2h.main
        Catch2::Catch2
    )
    target_include_directories(${test_target} PUBLIC "${CUB_SOURCE_DIR}/test")

    add_test(
      NAME ${test_target}
      # gersemi: off
      COMMAND
        "${CMAKE_COMMAND}"
           "-DCCCL_SOURCE_DIR=${CCCL_SOURCE_DIR}"
           "-DTEST=$<TARGET_FILE:${test_target}>"
           "-DTYPE=Catch2"
           -P "${CUB_SOURCE_DIR}/test/run_test.cmake"
      # gersemi: on
    )
    set_tests_properties(
      ${test_target}
      PROPERTIES SKIP_REGULAR_EXPRESSION "CCCL_SKIP_TEST"
    )

    if ("${test_target}" MATCHES "nvrtc")
      configure_file(
        "cmake/nvrtc_args.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/nvrtc_args.h"
      )
      target_include_directories(
        ${test_target}
        PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
      )
    endif()

    if ("${test_src}" MATCHES "test_iterator\\.cu$")
      target_compile_options(${test_target} PRIVATE -ftemplate-depth=1000) # for handling large type lists
    endif()

    # enable lambdas for all API examples
    if ("${test_src}" MATCHES "test.+_api\\.cu$")
      target_compile_options(
        ${test_target}
        PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
      )
    endif()

    if ("${test_src}" MATCHES "test_device_segmented_scan_api\\.cu$")
      if (
        "Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13
      )
        # When clang >= 13 is used as host compiler, we get the following warning:
        #   nvcc_internal_extended_lambda_implementation:312:22: error: definition of implicit copy constructor for '__nv_hdl_wrapper_t<false, true, false, __nv_dl_tag<void (*)(), &TestAddressStabilityLambda, 2>, int (const int &)>' is deprecated because it has a user-declared copy assignment operator [-Werror,-Wdeprecated-copy]
        #   312 | __nv_hdl_wrapper_t & operator=(const __nv_hdl_wrapper_t &in) = delete;
        #   |                      ^
        # Let's suppress it until NVBug 4980157 is resolved.
        target_compile_options(
          ${test_target}
          PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: -Wno-deprecated-copy>
        )
      endif()
    endif()
  else() # Not catch2:
    cccl_add_executable(
      ${test_target}
      SOURCES "${test_src}"
      NO_METATARGETS # Added manually for non-fail tests
    )
    target_link_libraries(
      ${test_target}
      PRIVATE #
        cub.compiler_interface
        cccl.c2h
    )
    target_include_directories(${test_target} PRIVATE "${CUB_SOURCE_DIR}/test")
    target_compile_definitions(${test_target} PRIVATE CUB_DEBUG_SYNC)

    if ("${test_target}" MATCHES "nvtx_in_usercode")
      target_link_libraries(${test_target} PRIVATE nvtx3-cpp)
    endif()

    _cub_is_fail_test(is_fail_test "${test_src}")
    if (is_fail_test)
      cccl_add_xfail_compile_target_test(
        ${test_target}
        SOURCE_FILE "${test_src}"
        ERROR_REGEX_LABEL "expected-error"
        ERROR_NUMBER_TARGET_NAME_REGEX "\\.err_([0-9]+)"
      )
    else()
      cccl_ensure_metatargets(${test_target} METATARGET_PATH ${metatarget_path})

      add_test(
        NAME ${test_target}
        # gersemi: off
        COMMAND
          "${CMAKE_COMMAND}"
            "-DCCCL_SOURCE_DIR=${CCCL_SOURCE_DIR}"
            "-DTEST=$<TARGET_FILE:${test_target}>"
            -P "${CUB_SOURCE_DIR}/test/run_test.cmake"
        # gersemi: on
      )
      set_tests_properties(
        ${test_target}
        PROPERTIES SKIP_REGULAR_EXPRESSION "CCCL_SKIP_TEST"
      )
    endif()
  endif() # Not catch2 test

  # Ensure that we test with assertions enabled
  target_compile_definitions(${test_target} PRIVATE CCCL_ENABLE_ASSERTIONS)
endfunction()

# Sets out_var to launch id if the label contains launch variants
function(_cub_has_lid_variant out_var label)
  string(FIND "${label}" "lid_" idx)
  if (idx EQUAL -1)
    set(${out_var} 0 PARENT_SCOPE)
  else()
    set(${out_var} 1 PARENT_SCOPE)
  endif()
endfunction()

# Sets out_var to 1 if the label contains "lid_1", e.g. launch id corresponds
# to device-side (CDP) launch.
function(_cub_launcher_id out_var label)
  string(REGEX MATCH "lid_([0-9]+)" MATCH_RESULT "${label}")
  if (MATCH_RESULT)
    set(${out_var} ${CMAKE_MATCH_1} PARENT_SCOPE)
  else()
    set(${out_var} 0 PARENT_SCOPE)
  endif()
endfunction()

foreach (test_src IN LISTS test_srcs)
  get_filename_component(test_name "${test_src}" NAME_WE)
  string(REGEX REPLACE "^catch2_test_" "" test_name "${test_name}")
  string(REGEX REPLACE "^test_" "" test_name "${test_name}")

  # Group sets of tests into metatargets based on their prefixes:
  string(REGEX REPLACE "^thread_" "thread." test_name "${test_name}")
  string(REGEX REPLACE "^warp_" "warp." test_name "${test_name}")
  string(REGEX REPLACE "^block_" "block." test_name "${test_name}")
  string(REGEX REPLACE "^device_" "device." test_name "${test_name}")
  string(REGEX REPLACE "^util_" "util." test_name "${test_name}")

  if ("${test_name}" MATCHES "nvrtc")
    if (NOT build_nvrtc_tests)
      continue()
    endif()
  endif()

  cccl_parse_variant_params(
    "${test_src}"
    num_variants
    variant_labels
    variant_defs
  )

  if (num_variants EQUAL 0)
    if (${CUB_FORCE_RDC})
      set(launcher 1)
    else()
      set(launcher 0)
    endif()

    # FIXME: There are a few remaining device algorithm tests that have not been ported to
    # use Catch2 and lid variants. Mark these as `lid_0/1` so they'll run in the appropriate
    # CI configs:
    set(variant_suffix)
    string(REGEX MATCH "^device\\." is_device_test "${test_name}")
    _cub_is_fail_test(is_fail_test "%{test_name}")
    if (is_device_test AND NOT is_fail_test)
      string(APPEND variant_suffix ".lid_${launcher}")
    endif()

    # Only one version of this test.
    cub_add_test(test_target ${test_name} "${variant_suffix}" "${test_src}" ${launcher})
    cub_configure_cuda_target(${test_target} RDC ${CUB_FORCE_RDC})
  else() # has variants:
    cccl_log_variant_params(
      "${test_name}"
      ${num_variants}
      variant_labels
      variant_defs
    )

    # Subtract 1 to support the inclusive endpoint of foreach(...RANGE...):
    math(EXPR range_end "${num_variants} - 1")

    # Generate multiple tests, one per variant.
    foreach (var_idx RANGE ${range_end})
      cccl_get_variant_data(variant_labels variant_defs ${var_idx} label defs)
      set(variant_suffix ".${label}")

      # If a `label` is `lid`, it is assumed that the parameter is used to explicitly
      # test variants built with different launchers. The `values` for such a
      # parameter must be `0:1:2`, with:
      # - `0` indicating host launch and CDP disabled (RDC off),
      # - `1` indicating device launch and CDP enabled (RDC on),
      # - `2` indicating graph capture launch and CDP disabled (RDC off).
      #
      # Tests that do not contain a variant labeled `lid` will only enable RDC if
      # the CMake config enables them.
      _cub_has_lid_variant(explicit_launcher "${label}")
      _cub_launcher_id(explicit_launcher_id "${label}")

      if (${explicit_launcher})
        set(launcher_id "${explicit_launcher_id}")
      else()
        if (${CUB_FORCE_RDC})
          set(launcher_id 1)
        else()
          set(launcher_id 0)
        endif()
      endif()

      _cub_launcher_requires_rdc(cdp_val "${launcher_id}")

      if (cdp_val AND NOT CUB_ENABLE_RDC_TESTS)
        continue()
      endif()

      if (NOT launcher_id EQUAL 0 AND NOT CUB_ENABLE_LAUNCH_VARIANTS)
        continue()
      endif()

      cub_add_test(test_target ${test_name} ${variant_suffix} "${test_src}" ${launcher_id})
      target_compile_definitions(${test_target} PRIVATE ${defs})

      # Enable RDC if the test either:
      # 1. Explicitly requests it (lid_1 label)
      # 2. Does not have an explicit CDP variant (no lid_0, lid_1, or lid_2) but
      #    RDC testing is forced
      #
      # Tests that explicitly request no cdp (lid_0 label) should never enable
      # RDC.
      cub_configure_cuda_target(${test_target} RDC ${cdp_val})
    endforeach() # Variant
  endif() # Has variants
endforeach() # Source file

add_subdirectory(cmake)
add_subdirectory(ptx-json)
