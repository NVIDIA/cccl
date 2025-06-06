# Create meta targets that build all tests for a single configuration:
foreach(thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)
  set(config_meta_target ${config_prefix}.tests)
  add_custom_target(${config_meta_target})
  add_dependencies(${config_prefix}.all ${config_meta_target})
endforeach()

# Generate testing framework libraries:
add_subdirectory(unittest)

cccl_get_catch2()

# Some tests only support certain host.device configurations. Use this macro to
# declare allowed configurations. If not specified, all host.device config are
# used.
set(restricted_tests)
macro(thrust_declare_test_restrictions test_name)
  list(APPEND restricted_tests ${test_name})
  list(APPEND ${test_name}_host.device_allowed ${ARGN})
endmacro()

# This test is incompatible with TBB and OMP, since it requires special per-device
# handling to process exceptions in a device function, which is only implemented
# for CUDA.
thrust_declare_test_restrictions(unittest_static_assert CPP.CPP CPP.CUDA)

# In the TBB backend, reduce_by_key does not currently work with transform_output_iterator
# https://github.com/NVIDIA/thrust/issues/1811
thrust_declare_test_restrictions(transform_output_iterator_reduce_by_key CPP.CPP CPP.OMP CPP.CUDA)

function(thrust_is_catch2_test result_var test_src)
  # If the test_src contains the substring "catch2_test_", `result_var` will be set to TRUE.
  string(FIND "${test_src}" "catch2_test_" idx)
  if (idx EQUAL -1)
    set(${result_var} FALSE PARENT_SCOPE)
  else()
    set(${result_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

## thrust_add_test
#
# Add a test executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the test
#   target. Useful for post-processing target information per-backend.
# test_name: The name of the test minus "<config_prefix>.test." For example,
#   testing/vector.cu will be "vector", and testing/cuda/copy.cu will be
#   "cuda.copy".
# test_src: The source file that implements the test.
# thrust_target: The reference thrust target with configuration information.
#
function(thrust_add_test target_name_var test_name test_src thrust_target)
  thrust_get_target_property(config_host ${thrust_target} HOST)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

  thrust_is_catch2_test(is_catch2_test "${test_src}")

  # Wrap the .cu file in .cpp for non-CUDA backends
  if ("CUDA" STREQUAL "${config_device}")
    set(real_test_src "${test_src}")
  else()
    thrust_wrap_cu_in_cpp(real_test_src "${test_src}" ${thrust_target})
  endif()

  # The actual name of the test's target:
  set(test_target ${config_prefix}.test.${test_name})
  set(${target_name_var} ${test_target} PARENT_SCOPE)

  # Related target names:
  set(config_framework_target ${config_prefix}.test.framework)
  set(config_meta_target ${config_prefix}.tests)
  set(test_meta_target thrust.all.test.${test_name})

  add_executable(${test_target} "${real_test_src}")

  if (is_catch2_test)
    target_link_libraries(${test_target} PRIVATE ${thrust_target} Catch2::Catch2WithMain)
  else()
    target_link_libraries(${test_target} PRIVATE ${config_framework_target})
  endif()
  target_include_directories(${test_target} PRIVATE "${Thrust_SOURCE_DIR}/testing")
  thrust_clone_target_properties(${test_target} ${thrust_target})

  if (NOT "Clang" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
    target_compile_definitions(${test_target} PRIVATE THRUST_TEST_DEVICE_SIDE)
  endif()

  # Ensure that we test with assertions enabled
  target_compile_definitions(${test_target} PRIVATE CCCL_ENABLE_ASSERTIONS)

  # Add to the active configuration's meta target
  add_dependencies(${config_meta_target} ${test_target})

  # Meta target that builds tests with this name for all configurations:
  if (NOT TARGET ${test_meta_target})
    add_custom_target(${test_meta_target})
  endif()
  add_dependencies(${test_meta_target} ${test_target})

  if (is_catch2_test)
    add_test(NAME ${test_target} COMMAND "$<TARGET_FILE:${test_target}>")
  else()
    add_test(NAME ${test_target}
      COMMAND "${CMAKE_COMMAND}"
      "-DTHRUST_BINARY=$<TARGET_FILE:${test_target}>"
      "-DTHRUST_SOURCE=${Thrust_SOURCE_DIR}"
      -P "${Thrust_SOURCE_DIR}/cmake/ThrustRunTest.cmake"
    )
  endif()

  # Run OMP/TBB tests in serial. Multiple OMP processes will massively
  # oversubscribe the machine with GCC's OMP, and we want to test these with
  # the full CPU available to each unit test.
  set(config_systems ${config_host} ${config_device})
  if (("OMP" IN_LIST config_systems) OR ("TBB" IN_LIST config_systems))
    set_tests_properties(${test_target} PROPERTIES RUN_SERIAL ON)
  endif()

  # Check for per-test script. Script will be included in the current scope
  # to allow custom property modifications.
  get_filename_component(test_cmake_script "${test_src}" NAME_WLE)
  set(test_cmake_script "${CMAKE_CURRENT_LIST_DIR}/${test_cmake_script}.cmake")
  # Use a glob so we can detect if this changes:
  file(GLOB test_cmake_script
    RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
    CONFIGURE_DEPENDS
    "${test_cmake_script}"
  )
  if (test_cmake_script) # Will be non-empty only if the script exists
    include("${test_cmake_script}")
  endif()
endfunction()

file(GLOB test_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  *.cu *.cpp
)

# Add common tests to all configs:
foreach(thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_host ${thrust_target} HOST)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

  foreach(test_src IN LISTS test_srcs)
    get_filename_component(test_name "${test_src}" NAME_WLE)
    string(REGEX REPLACE "^catch2_test_" "" test_name "${test_name}")

    # Is this test restricted to only certain host/device combinations?
    if(${test_name} IN_LIST restricted_tests)
      # Is the current host/device combination supported?
      if (NOT "${config_host}.${config_device}" IN_LIST
            ${test_name}_host.device_allowed)
        continue()
      endif()
    endif()

    thrust_add_test(test_target ${test_name} "${test_src}" ${thrust_target})

    if ("CUDA" STREQUAL "${config_device}")
      thrust_configure_cuda_target(${test_target} RDC ${THRUST_FORCE_RDC})
    endif()
  endforeach()
endforeach()

# Add specialized tests:
add_subdirectory(cmake)
add_subdirectory(cpp)
add_subdirectory(cuda)
add_subdirectory(omp)
