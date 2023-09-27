if(CMAKE_GENERATOR MATCHES "^Visual Studio")
  if(CUB_ENABLE_RDC_TESTS)
    if("${CMAKE_VERSION}" VERSION_LESS 3.27.5)
      # https://gitlab.kitware.com/cmake/cmake/-/merge_requests/8794
      message(WARNING "CMake 3.27.5 or newer is required to enable RDC tests in Visual Studio.")
      cmake_minimum_required(VERSION 3.27.5)
    endif()
  endif()
endif()

if ("NVHPC" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  # NVBugs 200770766
  set(CUB_SEPARATE_CATCH2 ON)
else()
  option(CUB_SEPARATE_CATCH2
    "Build each catch2 test as a separate executable."
    OFF
  )
endif()

include("${CUB_SOURCE_DIR}/cmake/CPM.cmake")
CPMAddPackage("gh:catchorg/Catch2@2.13.9")

option(METAL_BUILD_DOC OFF)
option(METAL_BUILD_EXAMPLES OFF)
option(METAL_BUILD_TESTS OFF)
CPMAddPackage("gh:brunocodutra/metal@2.1.4")

find_package(CUDAToolkit)

set(curand_default OFF)
if (CUDA_curand_LIBRARY)
  set(curand_default ON)
endif()

option(CUB_C2H_ENABLE_CURAND "Use CUDA CURAND library" ${curand_default})

# The function below reads the filepath `src`, extracts the %PARAM% comments,
# and fills `labels_var` with a list of `label1_value1.label2_value2...`
# strings, and puts the corresponding `DEFINITION=value1:DEFINITION=value2`
# entries into `defs_var`.
#
# See the README.md file in this directory for background info.
function(cub_get_test_params src labels_var defs_var)
  file(READ "${src}" file_data)
  set(param_regex "//[ ]+%PARAM%[ ]+([^ ]+)[ ]+([^ ]+)[ ]+([^\n]*)")

  string(REGEX MATCHALL
    "${param_regex}"
    matches
    "${file_data}"
  )

  set(variant_labels)
  set(variant_defs)

  foreach(match IN LISTS matches)
    string(REGEX MATCH
      "${param_regex}"
      unused
      "${match}"
    )

    set(def ${CMAKE_MATCH_1})
    set(label ${CMAKE_MATCH_2})
    set(values "${CMAKE_MATCH_3}")
    string(REPLACE ":" ";" values "${values}")

    # Build lists of test name suffixes (labels) and preprocessor definitions
    # (defs) containing the cartesian product of all param values:
    if (NOT variant_labels)
      foreach(value IN LISTS values)
        list(APPEND variant_labels ${label}_${value})
      endforeach()
    else()
      set(tmp_labels)
      foreach(old_label IN LISTS variant_labels)
        foreach(value IN LISTS values)
          list(APPEND tmp_labels ${old_label}.${label}_${value})
        endforeach()
      endforeach()
      set(variant_labels "${tmp_labels}")
    endif()

    if (NOT variant_defs)
      foreach(value IN LISTS values)
        list(APPEND variant_defs ${def}=${value})
      endforeach()
    else()
      set(tmp_defs)
      foreach(old_def IN LISTS variant_defs)
        foreach(value IN LISTS values)
          list(APPEND tmp_defs ${old_def}:${def}=${value})
        endforeach()
      endforeach()
      set(variant_defs "${tmp_defs}")
    endif()
  endforeach()

  set(${labels_var} "${variant_labels}" PARENT_SCOPE)
  set(${defs_var} "${variant_defs}" PARENT_SCOPE)
endfunction()

# Create meta targets that build all tests for a single configuration:
foreach(cub_target IN LISTS CUB_TARGETS)
  cub_get_target_property(config_prefix ${cub_target} PREFIX)
  set(config_meta_target ${config_prefix}.tests)
  add_custom_target(${config_meta_target})
  add_dependencies(${config_prefix}.all ${config_meta_target})
endforeach()

file(GLOB test_srcs
  RELATIVE "${CUB_SOURCE_DIR}/test"
  CONFIGURE_DEPENDS
  test_*.cu
  catch2_test_*.cu
)

## cub_is_catch2_test
#
# If the test_src contains the substring "catch2_test_", `result_var` will
# be set to TRUE.
function(cub_is_catch2_test result_var test_src)
  string(FIND "${test_src}" "catch2_test_" idx)
  if (idx EQUAL -1)
    set(${result_var} FALSE PARENT_SCOPE)
  else()
    set(${result_var} TRUE PARENT_SCOPE)
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
function(cub_add_test target_name_var test_name test_src cub_target cdp)
  cub_get_target_property(config_prefix ${cub_target} PREFIX)

  cub_is_catch2_test(is_catch2_test "${test_src}")

  # The actual name of the test's target:
  set(test_target ${config_prefix}.test.${test_name})
  set(${target_name_var} ${test_target} PARENT_SCOPE)

  set(config_meta_target ${config_prefix}.tests)

  if (${cdp})
    set(cdp_val 1)
  else()
    set(cdp_val 0)
  endif()

  if (is_catch2_test)
    # Per config helper library:
    set(config_c2h_target ${config_prefix}.test.catch2_helper.cdp_${cdp_val})
    if (NOT TARGET ${config_c2h_target})
      add_library(${config_c2h_target} STATIC c2h/generators.cu)
      target_include_directories(${config_c2h_target} PUBLIC "${CUB_SOURCE_DIR}/test")
      cub_clone_target_properties(${config_c2h_target} ${cub_target})
      cub_configure_cuda_target(${config_c2h_target} RDC ${cdp_val})
      target_link_libraries(${config_c2h_target} PRIVATE ${cub_target})
      if (CUB_C2H_ENABLE_CURAND)
        target_link_libraries(${config_c2h_target} PRIVATE CUDA::curand)
        target_compile_definitions(${config_c2h_target} PRIVATE C2H_HAS_CURAND=1)
      else()
        target_compile_definitions(${config_c2h_target} PRIVATE C2H_HAS_CURAND=0)
      endif()

      if (CUB_IN_THRUST)
        thrust_fix_clang_nvcc_build_for(${config_c2h_target})
      endif()
    endif() # config_c2h_target

    if (CUB_SEPARATE_CATCH2)
      add_executable(${test_target} "${test_src}")
      target_compile_definitions(${test_target} PRIVATE "CUB_CONFIG_MAIN")
      add_dependencies(${config_meta_target} ${test_target})

      add_test(NAME ${test_target} COMMAND "$<TARGET_FILE:${test_target}>")
    else() # Not CUB_SEPARATE_CATCH2
      # Per config catch2 runner
      set(config_c2run_target ${config_prefix}.catch2_test.cdp_${cdp_val})
      if (NOT TARGET ${config_c2run_target})
        add_executable(${config_c2run_target} catch2_runner.cpp catch2_runner_helper.cu)
        target_link_libraries(${config_c2run_target} PRIVATE
          ${cub_target}
          ${config_c2h_target}
          Metal
          Catch2::Catch2)
        cub_clone_target_properties(${config_c2run_target} ${cub_target})
        cub_configure_cuda_target(${config_c2run_target} RDC ${cdp_val})
        add_dependencies(${config_meta_target} ${config_c2run_target})
        target_include_directories(${config_c2run_target} PRIVATE
          "${CUB_SOURCE_DIR}/test"
        )
        if ("NVHPC" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
          target_link_options(${config_c2run_target} PRIVATE "-cuda")
        endif()

        if (CUB_IN_THRUST)
          thrust_fix_clang_nvcc_build_for(${config_c2run_target})
        endif()

        add_test(NAME ${config_c2run_target}
          COMMAND "$<TARGET_FILE:${config_c2run_target}>"
        )
      endif() # per config catch2 runner

      add_library(${test_target} OBJECT "${test_src}")

      if(CMAKE_GENERATOR MATCHES "^Visual Studio")
        target_link_libraries(${config_c2run_target} PRIVATE $<TARGET_OBJECTS:${test_target}>)
      else()
        target_link_libraries(${config_c2run_target} PRIVATE ${test_target})
      endif()
      
    endif() # CUB_SEPARATE_CATCH2

    if (CUB_IN_THRUST)
      thrust_fix_clang_nvcc_build_for(${test_target})
    endif()

    target_link_libraries(${test_target} PRIVATE
      ${cub_target}
      ${config_c2h_target}
      Metal
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
    target_link_libraries(${test_target} ${cub_target})
    cub_clone_target_properties(${test_target} ${cub_target})
    target_include_directories(${test_target} PRIVATE "${CUB_SOURCE_DIR}/test")
    target_compile_definitions(${test_target} PRIVATE CUB_DETAIL_DEBUG_ENABLE_SYNC)

    if (CUB_IN_THRUST)
      thrust_fix_clang_nvcc_build_for(${test_target})
    endif()

    # Add to the active configuration's meta target
    add_dependencies(${config_meta_target} ${test_target})

    # Meta target that builds tests with this name for all configurations:
    if (NOT TARGET ${test_meta_target})
      add_custom_target(${test_meta_target})
    endif()
    add_dependencies(${test_meta_target} ${test_target})

    add_test(NAME ${test_target} COMMAND "$<TARGET_FILE:${test_target}>")
  endif() # Not catch2 test
endfunction()

# Sets out_var to 1 if the label contains cdp variants, regardless of whether
# or not CDP is enabled in this particular variant.
function(_cub_has_cdp_variant out_var label)
  string(FIND "${label}" "cdp_" idx)
  if (idx EQUAL -1)
    set(${out_var} 0 PARENT_SCOPE)
  else()
    set(${out_var} 1 PARENT_SCOPE)
  endif()
endfunction()

# Sets out_var to 1 if the label contains "cdp_1", e.g. cdp is explicitly
# requested for this variant.
function(_cub_is_cdp_enabled_variant out_var label)
  string(FIND "${label}" "cdp_1" idx)
  if (idx EQUAL -1)
    set(${out_var} 0 PARENT_SCOPE)
  else()
    set(${out_var} 1 PARENT_SCOPE)
  endif()
endfunction()

foreach (test_src IN LISTS test_srcs)
  get_filename_component(test_name "${test_src}" NAME_WE)
  string(REGEX REPLACE "^catch2_test_" "" test_name "${test_name}")
  string(REGEX REPLACE "^test_" "" test_name "${test_name}")

  cub_get_test_params("${test_src}" variant_labels variant_defs)
  list(LENGTH variant_labels num_variants)

  # Subtract 1 to support the inclusive endpoint of foreach(...RANGE...):
  math(EXPR range_end "${num_variants} - 1")

  # Verbose output:
  if (num_variants GREATER 0)
    message(VERBOSE "Detected ${num_variants} variants of test '${test_src}':")
    foreach(var_idx RANGE ${range_end})
      math(EXPR i "${var_idx} + 1")
      list(GET variant_labels ${var_idx} label)
      list(GET variant_defs ${var_idx} defs)
      message(VERBOSE "  ${i}: ${test_name} ${label} ${defs}")
    endforeach()
  endif()

  foreach(cub_target IN LISTS CUB_TARGETS)
    cub_get_target_property(config_prefix ${cub_target} PREFIX)

    if (num_variants EQUAL 0)
      # Only one version of this test.
      cub_add_test(test_target ${test_name} "${test_src}" ${cub_target} ${CUB_FORCE_RDC})
      cub_configure_cuda_target(${test_target} RDC ${CUB_FORCE_RDC})
    else() # has variants:
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

      # Generate multiple tests, one per variant.
      # See `cub_get_test_params` for details.
      foreach(var_idx RANGE ${range_end})
        list(GET variant_labels ${var_idx} label)
        list(GET variant_defs ${var_idx} defs)
        string(REPLACE ":" ";" defs "${defs}")
        # A unique index per variant:
        list(APPEND defs VAR_IDX=${var_idx})

        # Check if the test has explicit CDP variants:
        _cub_has_cdp_variant(explicit_cdp "${label}")
        _cub_is_cdp_enabled_variant(enable_cdp "${label}")

        if (enable_cdp AND NOT CUB_ENABLE_RDC_TESTS)
          continue()
        endif()

        # Enable RDC if the test either:
        # 1. Explicitly requests it (cdp_1 label)
        # 2. Does not have an explicit CDP variant (no cdp_0 or cdp_1) but
        #    RDC testing is forced
        #
        # Tests that explicitly request no cdp (cdp_0 label) should never enable
        # RDC.
        if (explicit_cdp)
          set(cdp_val ${enable_cdp})
        else()
          set(cdp_val ${CUB_FORCE_RDC})
        endif()

        cub_add_test(test_target
          ${test_name}.${label}
          "${test_src}"
          ${cub_target}
          ${cdp_val})
        cub_configure_cuda_target(${test_target} RDC ${cdp_val})
        add_dependencies(${variant_meta_target} ${test_target})
        add_dependencies(${cub_variant_meta_target} ${test_target})
        target_compile_definitions(${test_target} PRIVATE ${defs})
      endforeach() # Variant
    endif() # Has variants
  endforeach() # CUB targets
endforeach() # Source file

add_subdirectory(cmake)
