option(CUB_ENABLE_LAUNCH_VARIANTS "Enable CUB launch variants (lid_1 and lid_2)" ON)

if(CMAKE_GENERATOR MATCHES "^Visual Studio")
  if(CUB_ENABLE_RDC_TESTS)
    if("${CMAKE_VERSION}" VERSION_LESS 3.27.5)
      # https://gitlab.kitware.com/cmake/cmake/-/merge_requests/8794
      message(WARNING "CMake 3.27.5 or newer is required to enable RDC tests in Visual Studio.")
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

# Create meta targets that build all tests for a single configuration:
foreach(cub_target IN LISTS CUB_TARGETS)
  cub_get_target_property(config_prefix ${cub_target} PREFIX)
  set(config_meta_target ${config_prefix}.tests)
  add_custom_target(${config_meta_target})
  add_dependencies(${config_prefix}.all ${config_meta_target})
endforeach()

file(GLOB_RECURSE test_srcs
  RELATIVE "${CUB_SOURCE_DIR}/test"
  CONFIGURE_DEPENDS
  test_*.cu
  catch2_test_*.cu
)

# nvtx headers contain a variable named `module`, which breaks nvc++ as that is a keyword
if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}" AND NOT "${CMAKE_CXX_STANDARD}" MATCHES "17")
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
#   target. Useful for post-processing target information.
# test_name: The name of the test minus "<config_prefix>.test." For example,
#   testing/vector.cu will be "vector", and testing/cuda/copy.cu will be
#   "cuda.copy".
# test_src: The source file that implements the test.
# cub_target: The reference cub target with configuration information.
#
function(cub_add_test target_name_var test_name test_src cub_target launcher_id)
  cub_get_target_property(config_prefix ${cub_target} PREFIX)

  _cub_is_catch2_test(is_catch2_test "${test_src}")
  _cub_launcher_requires_rdc(cdp_val "${launcher_id}")

  # The actual name of the test's target:
  set(test_target ${config_prefix}.test.${test_name})
  set(${target_name_var} ${test_target} PARENT_SCOPE)

  set(config_meta_target ${config_prefix}.tests)

  if (is_catch2_test)
    # Per config helper library:
    set(config_c2h_target ${config_prefix}.test.catch2_helper.lid_${launcher_id})
    if (NOT TARGET ${config_c2h_target})
      add_library(${config_c2h_target} INTERFACE)
      target_include_directories(${config_c2h_target} INTERFACE "${CUB_SOURCE_DIR}/test")
      cub_clone_target_properties(${config_c2h_target} ${cub_target})
      cub_configure_cuda_target(${config_c2h_target} RDC ${cdp_val})
      target_link_libraries(${config_c2h_target} INTERFACE
        ${cub_target}
        cccl.c2h
        CUDA::nvrtc
        CUDA::cuda_driver
      )
    endif() # config_c2h_target

    add_executable(${test_target} "${test_src}")
    target_link_libraries(${test_target} PRIVATE cccl.c2h.main)
    add_dependencies(${config_meta_target} ${test_target})

    add_test(NAME ${test_target} COMMAND
      "${CMAKE_COMMAND}"
      "-DCCCL_SOURCE_DIR=${CCCL_SOURCE_DIR}"
      "-DTEST=$<TARGET_FILE:${test_target}>"
      "-DTYPE=Catch2"
      -P "${CUB_SOURCE_DIR}/test/run_test.cmake"
    )
    set_tests_properties(${test_target} PROPERTIES SKIP_REGULAR_EXPRESSION "CCCL_SKIP_TEST")

    if ("${test_target}" MATCHES "nvrtc")
      configure_file("cmake/nvrtc_args.h.in" ${CMAKE_CURRENT_BINARY_DIR}/nvrtc_args.h)
      target_include_directories(${test_target} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if ("${test_target}" MATCHES "test.iterator")
      target_compile_options(${test_target} PRIVATE -ftemplate-depth=1000) # for handling large type lists
    endif()

    # enable lambdas for all API examples
    if ("${test_target}" MATCHES "test.[A-Za-z0-9_]+_api")
      target_compile_options(${test_target} PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>)
    endif()

    target_link_libraries(${test_target} PRIVATE
      ${cub_target}
      ${config_c2h_target}
      Catch2::Catch2
    )
    cub_clone_target_properties(${test_target} ${cub_target})
    target_include_directories(${test_target}
      PUBLIC "${CUB_SOURCE_DIR}/test"
    )
  else() # Not catch2:
    # Related target names:
    set(test_meta_target cub.all.test.${test_name})

    add_executable(${test_target} "${test_src}")
    target_link_libraries(${test_target} PRIVATE
      ${cub_target}
      cccl.c2h
    )
    cub_clone_target_properties(${test_target} ${cub_target})
    target_include_directories(${test_target} PRIVATE "${CUB_SOURCE_DIR}/test")
    target_compile_definitions(${test_target} PRIVATE CUB_DEBUG_SYNC)

    if ("${test_target}" MATCHES "nvtx_in_usercode")
      target_link_libraries(${test_target} PRIVATE nvtx3-cpp)
    endif()

    _cub_is_fail_test(is_fail_test "${test_src}")
    if (is_fail_test)
      set_target_properties(${test_target} PROPERTIES EXCLUDE_FROM_ALL true
                                           EXCLUDE_FROM_DEFAULT_BUILD true)
      add_test(NAME ${test_target}
               COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}"
                                        --target ${test_target}
                                        --config $<CONFIGURATION>)
      string(REGEX MATCH "err_([0-9]+)" MATCH_RESULT "${test_name}")
      file(READ ${test_src} test_content)
      if(MATCH_RESULT)
        string(REGEX MATCH "// expected-error-${CMAKE_MATCH_1}+ {{\"([^\"]+)\"}}" expected_errors_matches ${test_content})

        if (expected_errors_matches)
          set_tests_properties(${test_target} PROPERTIES PASS_REGULAR_EXPRESSION "${CMAKE_MATCH_1}")
        else()
          set_tests_properties(${test_target} PROPERTIES WILL_FAIL true)
        endif()
      else()
        string(REGEX MATCH "// expected-error {{\"([^\"]+)\"}}" expected_errors_matches ${test_content})

        if (expected_errors_matches)
          set_tests_properties(${test_target} PROPERTIES PASS_REGULAR_EXPRESSION "${CMAKE_MATCH_1}")
        else()
          set_tests_properties(${test_target} PROPERTIES WILL_FAIL true)
        endif()
      endif()
    else()
      # Add to the active configuration's meta target
      add_dependencies(${config_meta_target} ${test_target})

      # Meta target that builds tests with this name for all configurations:
      if (NOT TARGET ${test_meta_target})
        add_custom_target(${test_meta_target})
      endif()
      add_dependencies(${test_meta_target} ${test_target})

      add_test(NAME ${test_target} COMMAND
        "${CMAKE_COMMAND}"
        "-DCCCL_SOURCE_DIR=${CCCL_SOURCE_DIR}"
        "-DTEST=$<TARGET_FILE:${test_target}>"
        -P "${CUB_SOURCE_DIR}/test/run_test.cmake"
      )
      set_tests_properties(${test_target} PROPERTIES SKIP_REGULAR_EXPRESSION "CCCL_SKIP_TEST")
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
  if(MATCH_RESULT)
    set(${out_var} ${CMAKE_MATCH_1} PARENT_SCOPE)
  else()
    set(${out_var} 0 PARENT_SCOPE)
  endif()
endfunction()

foreach (test_src IN LISTS test_srcs)
  get_filename_component(test_name "${test_src}" NAME_WE)
  string(REGEX REPLACE "^catch2_test_" "" test_name "${test_name}")
  string(REGEX REPLACE "^test_" "" test_name "${test_name}")

  if ("${test_name}" MATCHES "nvrtc")
    if (NOT build_nvrtc_tests)
      continue()
    endif()
  endif()

  cccl_parse_variant_params("${test_src}" num_variants variant_labels variant_defs)

  foreach(cub_target IN LISTS CUB_TARGETS)
    cub_get_target_property(config_prefix ${cub_target} PREFIX)

    if (num_variants EQUAL 0)
      if (${CUB_FORCE_RDC})
        set(launcher 1)
      else()
        set(launcher 0)
      endif()

      # FIXME: There are a few remaining device algorithm tests that have not been ported to
      # use Catch2 and lid variants. Mark these as `lid_0/1` so they'll run in the appropriate
      # CI configs:
      string(REGEX MATCH "^device_" is_device_test "${test_name}")
      _cub_is_fail_test(is_fail_test "%{test_name}")
      if (is_device_test AND NOT is_fail_test)
        string(APPEND test_name ".lid_${launcher}")
      endif()

      # Only one version of this test.
      cub_add_test(test_target ${test_name} "${test_src}" ${cub_target} ${launcher})
      cub_configure_cuda_target(${test_target} RDC ${CUB_FORCE_RDC})
    else() # has variants:
      cccl_log_variant_params("${test_name}" ${num_variants} variant_labels variant_defs)

      # Meta target to build all parametrizations of the current test for the
      # current CUB_TARGET config
      set(variant_meta_target ${config_prefix}.test.${test_name}.all)
      if (NOT TARGET ${variant_meta_target})
        add_custom_target(${variant_meta_target})
      endif()

      # Meta target to build all parametrizations of the current test for all
      # CUB_TARGET configs
      set(cub_variant_meta_target cub.all.test.${test_name}.all)
      if (NOT TARGET ${cub_variant_meta_target})
        add_custom_target(${cub_variant_meta_target})
      endif()

      # Subtract 1 to support the inclusive endpoint of foreach(...RANGE...):
      math(EXPR range_end "${num_variants} - 1")

      # Generate multiple tests, one per variant.
      foreach(var_idx RANGE ${range_end})
        cccl_get_variant_data(variant_labels variant_defs ${var_idx} label defs)

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

        cub_add_test(test_target ${test_name}.${label} "${test_src}" ${cub_target} ${launcher_id})

        # Enable RDC if the test either:
        # 1. Explicitly requests it (lid_1 label)
        # 2. Does not have an explicit CDP variant (no lid_0, lid_1, or lid_2) but
        #    RDC testing is forced
        #
        # Tests that explicitly request no cdp (lid_0 label) should never enable
        # RDC.
        cub_configure_cuda_target(${test_target} RDC ${cdp_val})
        add_dependencies(${variant_meta_target} ${test_target})
        add_dependencies(${cub_variant_meta_target} ${test_target})
        target_compile_definitions(${test_target} PRIVATE ${defs})
      endforeach() # Variant
    endif() # Has variants
  endforeach() # CUB targets
endforeach() # Source file

add_subdirectory(cmake)
add_subdirectory(ptx-json)
