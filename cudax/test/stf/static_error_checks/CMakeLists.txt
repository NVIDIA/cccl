set(
  static_assert_tests
  exec_place_guard_data_place.cu
  static.cu
  task_prototype.cu
  stream_task_prototype.cu
  graph_task_prototype.cu
)

## cudax_add_stf_static_assert_test
#
# Adds an EXCLUDE_FROM_ALL build target for `source` and a ctest that
# triggers the compilation and checks for "static assertion" in the output.
#
# target_name_var: Variable name to overwrite with the name of the test
#   target. Useful for adding target information after creation.
# source: The source file for the test.
#
# Additional args are passed to cudax_stf_configure_target.
function(cudax_stf_add_static_assert_test target_name_var source)
  get_filename_component(filename ${source} NAME_WE)

  set(test_target cudax.tests.stf.error.${filename})

  cccl_add_executable(${test_target} SOURCES ${source} NO_METATARGETS)
  cudax_stf_configure_target(${test_target} ${ARGN})
  target_link_libraries(${test_target} PRIVATE cudax.compiler_interface)
  cccl_add_xfail_compile_target_test(
    ${test_target}
    ERROR_REGEX "static_assert|static assertion"
  )

  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

foreach (source IN LISTS static_assert_tests)
  cudax_stf_add_static_assert_test(test_target "${source}")
endforeach()
