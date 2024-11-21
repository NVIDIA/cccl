set(static_assert_tests
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
#   target. Useful for post-processing target information.
# source: The source file for the test.
# cn_target: The reference cudax target with configuration information.
# Additional args are passed to cudax_stf_configure_target.
function(cudax_stf_add_static_assert_test target_name_var source cn_target)
  cudax_get_target_property(config_dialect ${cn_target} DIALECT)
  cudax_get_target_property(config_prefix ${cn_target} PREFIX)

  get_filename_component(filename ${source} NAME_WE)

  set(test_target "${config_prefix}.tests.stf.error.${filename}")

  add_executable(${test_target} ${source})
  cccl_configure_target(${test_target} DIALECT ${config_dialect})
  cudax_clone_target_properties(${test_target} ${cn_target})
  cudax_stf_configure_target(${test_target} ${ARGN})
  set_target_properties(${test_target} PROPERTIES
    EXCLUDE_FROM_ALL true
    EXCLUDE_FROM_DEFAULT_BUILD true
  )

  add_test(NAME ${test_target}
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}"
                             --target ${test_target}
                             --config $<CONFIGURATION>
  )
  set_tests_properties(${test_target} PROPERTIES
    PASS_REGULAR_EXPRESSION "static_assert|static assertion"
  )

  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

foreach(cn_target IN LISTS cudax_TARGETS)
  foreach(source IN LISTS static_assert_tests)
    cudax_stf_add_static_assert_test(test_target "${source}" ${cn_target})
  endforeach()
endforeach()
