# Usage:
# cccl_add_ctest(test_name
#                [LABEL_PATH label1.label2.label3]
#                [LABEL_MARKER marker]
#                [LABEL_NAME]
#                [add_test options]
# )
#
# Options:
# test_name: The name of the test that will be used by CTest.
# add_test options: Additional options to pass to add_test.
# LABEL_PATH: A set of heirarchical labels delimited by periods. Each label will include all
#   previous labels. For example, `LABEL_PATH thrust.cpp_cuda.cpp17.test` will add the labels
#   `thrust`, `thrust.cpp_cuda`, `thrust.cpp_cuda.cpp17`, and `thrust.cpp_cuda.cpp17.test`.
# LABEL_MARKER: See cccl_get_label_path docs for usage and examples.
# LABEL_NAME: If set, the test name will be directly used as the label_path.
function(cccl_add_ctest test_name)
  set(options LABEL_NAME)
  set(oneValueArgs LABEL_PATH LABEL_MARKER)
  set(multiValueArgs)
  cmake_parse_arguments(CACT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(add_test_opts ${CACT_UNPARSED_ARGUMENTS})

  if (DEFINED CACT_LABEL_NAME)
    set(CACT_LABEL_PATH ${test_name})
  endif()

  if (DEFINED CACT_LABEL_MARKER)
    cccl_get_label_path(CACT_LABEL_PATH ${test_name} ${CACT_LABEL_MARKER})
  endif()

  set(labels)
  if (DEFINED CACT_LABEL_PATH)
    set(label_tmp)
    string(REPLACE "." ";" label_list ${CACT_LABEL_PATH})
    foreach (label IN LISTS label_list)
      list(APPEND label_tmp "${label}")
      string(REPLACE ";" "." label_str "${label_tmp}")
      list(APPEND labels "${label_str}")
    endforeach()
  endif()

  add_test(NAME ${test_name} ${add_test_opts})
  set_property(TEST ${test_name} PROPERTY LABELS "${labels}")
endfunction()

# Extract a label path from a target name. `marker` is the last label used.
#
# cccl_get_label_path(label_path cub.cpp17.test.reduce.foo.bar.baz "test")
# will set`label_path` to `cub.cpp17.test`.
#
# If `lid_[0-9]` appears after the test name, it will be appended to the label path:
# cccl_get_label_path(label_path cub.cpp17.test.reduce.foo.bar.lid_2.baz "test")
# will set `label_path` to `cub.cpp17.test.lid_2`.
#
# If the target name starts with `thrust.<host>.<device>.cpp<dialect>....`, the host/device
# strings will be combined to form a single "<host>_<device>" label:
# cccl_get_label_path(label_path thrust.cpp.cuda.cpp17.test.reduce.foo.bar.lid_2.baz "test")
# will set `label_path` to `thrust.cpp_cuda.cpp17.test.lid_2`.
function(cccl_get_label_path label_path_var test_name marker)
  # Combine thrust host/device to a single label, if present:
  if (test_name MATCHES "^thrust\\.[^.]+\\.[^.]+\\.cpp.+$")
    string(REGEX REPLACE "^(thrust\\.[^.]+)\\.(.+)$" "\\1_\\2" test_name ${test_name})
  endif()

  # Truncate the label path to only include a single label after 'test.':
  string(REGEX MATCH ".+\\.${marker}" label_path ${test_name})

  # Append any `lid_X` options:
  string(REGEX MATCH "lid_[0-9]+" launcher_id_str ${test_name})
  if (launcher_id_str)
    string(APPEND label_path ".${launcher_id_str}")
  endif()

  set(${label_path_var} ${label_path} PARENT_SCOPE)
endfunction()
