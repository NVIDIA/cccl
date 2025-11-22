# Locate all test sources
file(
  GLOB test_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  *.cu
  *.cpp
)

# Thrust's legacy unit test framework
# Defines thrust.<host>.<device>.test.framework targets.
add_subdirectory(unittest)

# Default Test Setup
add_library(thrust.test.interface.default INTERFACE)
target_compile_definitions(
  thrust.test.interface.default
  INTERFACE
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:THRUST_TEST_DEVICE_SIDE>
    CCCL_ENABLE_ASSERTIONS
)
target_include_directories(
  thrust.test.interface.default
  INTERFACE "${Thrust_SOURCE_DIR}/testing"
)

# Catch2 Setup
cccl_get_catch2()

function(_thrust_is_catch2_src out_var test_src)
  set(${out_var} FALSE PARENT_SCOPE)
  if (test_src MATCHES "(^|/)catch2_test_")
    set(${out_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

# Failure test setup
# Tests suffixed with `_fail` are expected to fail compilation and require
# customized setup.
function(_thrust_is_fail_test out_var test_name)
  set(${out_var} FALSE PARENT_SCOPE)
  if (test_name MATCHES "\\.fail(\\.|$)")
    set(${out_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

# Convert known 'group_' prefixes to "group." for target naming.
set(
  _common_prefixes
  binary_search
  functional
  functional.placeholders
  mr
  pair
  reduce
  set
  set.difference
  set.intersection
  set.symmetric_difference
  set.union
  sort
  stable_sort
  stable_sort.by_key
  tuple
  vector
  zip
  zip.iterator
)
function(_thrust_test_prefix_to_group test_name_var)
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

# Special cases: some tests only support certain host.device configurations.
function(_thrust_test_is_supported out_var test_src thrust_target)
  thrust_get_target_property(config_host ${thrust_target} HOST)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  set(config_systems ${config_host} ${config_device})

  # Disabled for TBB: https://github.com/NVIDIA/thrust/issues/1811
  if (test_src MATCHES "^transform_output_iterator_reduce_by_key.cu")
    if ("TBB" IN_LIST config_systems)
      set(${out_var} FALSE PARENT_SCOPE)
      return()
    endif()
  endif()

  set(${out_var} TRUE PARENT_SCOPE)
endfunction()

# Special cases: Tweak build targets based on target name / file:
function(
  _thrust_test_build_target_special_case
  build_target
  test_src
  thrust_target
)
endfunction()

# Special cases: Tweak non-fail test targets based on target name / file:
function(
  _thrust_test_test_target_special_case
  test_target
  test_src
  thrust_target
)
  thrust_get_target_property(config_host ${thrust_target} HOST)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  set(config_systems ${config_host} ${config_device})

  # Run OMP/TBB tests in serial. Multiple OMP processes will massively
  # oversubscribe the machine with GCC's OMP, and we want to test these with
  # the full CPU available to each unit test.
  if (("OMP" IN_LIST config_systems) OR ("TBB" IN_LIST config_systems))
    set_tests_properties(${test_target} PROPERTIES RUN_SERIAL ON)
  endif()
endfunction()

#
# Generic test assembler:
#

function(thrust_add_test test_target test_src thrust_target)
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
    "thrust_add_test"
    self
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ERROR_UNPARSED
    DEFAULT_VALUES METATARGET_PATH "${test_target}"
  )
  thrust_get_target_property(config_host ${thrust_target} HOST)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)
  _thrust_is_catch2_src(is_catch2_test "${test_src}")
  _thrust_is_fail_test(is_fail "${test_target}")
  set(config_systems ${config_host} ${config_device})

  # Wrap the .cu file in .cpp for non-CUDA backends
  set(orig_test_src "${test_src}")
  if (NOT config_device STREQUAL "CUDA")
    thrust_wrap_cu_in_cpp(test_src "${test_src}" ${thrust_target})
  endif()

  cccl_add_executable(
    ${test_target}
    SOURCES "${test_src}"
    NO_METATARGETS # Added manually for non-fail tests
  )

  target_link_libraries(
    ${test_target}
    PRIVATE #
      ${thrust_target}
      thrust.test.interface.default
  )

  if (is_catch2_test)
    target_link_libraries(${test_target} PRIVATE Catch2::Catch2WithMain)
  else()
    target_link_libraries(
      ${test_target}
      PRIVATE ${config_prefix}.test.framework
    )
  endif()

  if ("CUDA" STREQUAL "${config_device}")
    cccl_set_rdc(${test_target} ${THRUST_FORCE_RDC})
  endif()

  _thrust_test_build_target_special_case(
    ${test_target}
    "${orig_test_src}"
    ${thrust_target}
  )

  # Fail tests are handled specially:
  if (is_fail)
    cccl_add_xfail_compile_target_test(
      ${test_target}
      # Use the original for caching between variants/configs
      SOURCE_FILE "${orig_test_src}"
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

  add_test(NAME ${test_target} COMMAND ${test_target})
  _thrust_test_test_target_special_case(
    ${test_target}
    "${orig_test_src}"
    ${thrust_target}
  )

  # Check for per-test script. Script will be included in the current scope
  # to allow custom property modifications.
  get_filename_component(test_cmake_script "${orig_test_src}" NAME_WLE)
  set(test_cmake_script "${CMAKE_CURRENT_LIST_DIR}/${test_cmake_script}.cmake")
  # Use a glob so we can detect if this changes:
  file(
    GLOB test_cmake_script
    RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
    CONFIGURE_DEPENDS
    "${test_cmake_script}"
  )
  if (test_cmake_script) # Will be non-empty only if the script exists
    include("${test_cmake_script}")
  endif()
endfunction()

# Create tests from a source file, handling variants.
# All created tests are returned in test_targets_var.
function(thrust_add_test_src test_targets_var test_src thrust_target)
  thrust_get_target_property(config_host ${thrust_target} HOST)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

  # Prepare the test_name
  get_filename_component(test_name "${test_src}" NAME_WLE)
  string(REGEX REPLACE "^catch2_test_" "" test_name "${test_name}")
  # Format known group prefixes in test names:
  _thrust_test_prefix_to_group(test_name ${_common_prefixes})
  string(PREPEND test_name "${config_prefix}.test.")

  # Detect %PARAM% variants:
  cccl_parse_variant_params(
    "${test_src}"
    num_variants
    variant_labels
    variant_defs
  )

  if (num_variants EQUAL 0)
    thrust_add_test(${test_name} "${test_src}" ${thrust_target})
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

    set(variant_name "${test_name}.${label}")
    thrust_add_test(
      ${variant_name}
      "${test_src}"
      ${thrust_target}
      # Don't create metatargets for variable labels, just the base testname:
      METATARGET_PATH "${test_name}"
    )
    target_compile_definitions(${variant_name} PRIVATE ${defs})

    list(APPEND test_targets ${variant_name})
  endforeach() # Variant

  set(${test_targets_var} "${test_targets}" PARENT_SCOPE)
endfunction()

foreach (thrust_target IN LISTS THRUST_TARGETS)
  foreach (test_src IN LISTS test_srcs)
    _thrust_test_is_supported(is_supported "${test_src}" ${thrust_target})
    if (is_supported)
      thrust_add_test_src(test_targets "${test_src}" ${thrust_target})
    endif()
  endforeach()
endforeach()

# TODO update the subdirs:
return()

# Add specialized tests:
add_subdirectory(cmake)
add_subdirectory(cpp)
add_subdirectory(cuda)
add_subdirectory(omp)
