# Thrust's legacy unit test framework:
# Defines thrust.<host>.<device>.test.framework targets.
add_subdirectory(unittest)

# All tests link to this:
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

#
# Customization points:
#

# If the test source is supported by the config:
#
function(_thrust_test_is_src_supported out_var test_src thrust_target)
  thrust_get_target_property(config_host ${thrust_target} HOST)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  set(config_systems ${config_host} ${config_device})

  set(result TRUE)

  # Per-backend tests:
  # gersemi: off
  if ((test_src MATCHES "^cpp/"  AND NOT config_device STREQUAL "CPP") OR
      (test_src MATCHES "^cuda/" AND NOT config_device STREQUAL "CUDA") OR
      (test_src MATCHES "^omp/"  AND NOT config_device STREQUAL "OMP") OR
      (test_src MATCHES "^tbb/"  AND NOT config_device STREQUAL "TBB"))
    # gersemi: on
    set(result FALSE)
  endif()

  # https://github.com/NVIDIA/thrust/issues/1811
  if (test_src MATCHES "^transform_output_iterator_reduce_by_key.cu")
    if ("TBB" IN_LIST config_systems)
      set(result FALSE)
    endif()
  endif()

  set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

# Customize the test source file (e.g. wrapping .cu in .cpp for non-CUDA backends):
#
function(_thrust_test_customize_src out_var test_src thrust_target)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)

  # Wrap .cu in .cpp for non-CUDA backends
  if (NOT config_device STREQUAL "CUDA")
    thrust_wrap_cu_in_cpp(test_src "${test_src}" ${thrust_target})
  endif()

  set(${out_var} "${test_src}" PARENT_SCOPE)
endfunction()

# Convert the source file path to a test case base name:
#
function(_thrust_test_src_to_basename out_var test_src thrust_target)
  get_filename_component(test_name "${test_src}" NAME_WLE)
  string(REGEX REPLACE "^catch2_test_" "" test_name "${test_name}")

  get_filename_component(test_dir "${test_src}" DIRECTORY)
  if (test_dir)
    string(REPLACE "/" "." test_name "${test_dir}.${test_name}")
  endif()

  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)
  string(PREPEND test_name "${config_prefix}.test.")

  set(${out_var} "${test_name}" PARENT_SCOPE)
endfunction()

# Link libraries, set defs, etc on the test executable target.
#
function(
  _thrust_test_customize_executable
  test_src
  test_name
  test_executable_target
  thrust_target
)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)
  _thrust_is_catch2_src(is_catch2 "${test_src}")

  thrust_add_all_config_metatarget(${test_executable_target})

  target_link_libraries(
    ${test_executable_target}
    PRIVATE #
      ${thrust_target}
      thrust.test.interface.default
  )

  if (is_catch2)
    target_link_libraries(
      ${test_executable_target}
      PRIVATE Catch2::Catch2WithMain
    )
  else()
    target_link_libraries(
      ${test_executable_target}
      PRIVATE ${config_prefix}.test.framework
    )
  endif()

  if (config_device STREQUAL "CUDA")
    cccl_set_rdc(${test_executable_target} ${THRUST_FORCE_RDC})
  endif()

  # gersemi: off
  if ((test_name MATCHES "test\\.address_stability") OR
      (test_name MATCHES "test\\.cuda\\.transform_iterator"))
    # gersemi: on
    target_compile_options(
      ${test_executable_target}
      PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
    )

    # NVBug 4980157: `__nv_hdl_wrapper_t` copy constructor deprecated warnings in Clang 13+
    # gersemi: off
    if ((CMAKE_CXX_COMPILER_ID STREQUAL Clang) AND
        (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13))
      # gersemi: on
      target_compile_options(
        ${test_executable_target}
        PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: -Wno-deprecated-copy>
      )
    endif()
  endif() # address_stability / cuda.transform_iterator

  if (test_name MATCHES "test\\.cuda\\.stream_per_thread")
    target_compile_options(
      ${test_executable_target}
      PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--default-stream=per-thread>
    )
  endif()
endfunction()

# Customize test properties such as SKIP, WILL_FAIL, etc.
#
function(
  _thrust_test_customize_test_properties
  test_src
  test_name
  test_executable_target
  ctest_name
  thrust_target
)
  thrust_get_target_property(config_host ${thrust_target} HOST)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  set(config_systems ${config_host} ${config_device})

  # Always skip tests that print CCCL_SKIP_TEST:
  set_tests_properties(
    ${ctest_name}
    PROPERTIES SKIP_REGULAR_EXPRESSION "CCCL_SKIP_TEST"
  )

  # Run OMP/TBB tests in serial to avoid oversubscription:
  if (("OMP" IN_LIST config_systems) OR ("TBB" IN_LIST config_systems))
    if (NOT test_name MATCHES "[_.]fail(\\.|$)")
      set_tests_properties(${ctest_name} PROPERTIES RUN_SERIAL ON)
    endif()
  endif()

  # Only nvcc supports --default-stream=per-thread.
  if (test_name MATCHES "cuda\\.stream_per_thread")
    if (NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
      set_tests_properties(${ctest_name} PROPERTIES WILL_FAIL ON)
    endif()
  endif()
endfunction()

#
# Add tests:
#

file(
  GLOB test_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  *.cu
  cpp/*.cu
  cuda/*.cu
  omp/*.cu
  tbb/*.cu
)

foreach (thrust_target IN LISTS THRUST_TARGETS)
  foreach (test_src IN LISTS test_srcs)
    cccl_add_tests_from_src(
      test_targets # unused
      "${test_src}"
      FUNC_IS_SRC_SUPPORTED _thrust_test_is_src_supported
      FUNC_CUSTOMIZE_SRC _thrust_test_customize_src
      FUNC_SRC_TO_BASENAME _thrust_test_src_to_basename
      FUNC_CUSTOMIZE_EXECUTABLE _thrust_test_customize_executable
      FUNC_CUSTOMIZE_TEST_PROPERTIES _thrust_test_customize_test_properties
      EXTRA_ARGN ${thrust_target}
    )
  endforeach()
endforeach()

add_subdirectory(cmake)
