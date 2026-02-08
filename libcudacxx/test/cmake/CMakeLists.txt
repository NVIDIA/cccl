# Check source code for issues that can be found by pattern matching:
add_test(
  NAME libcudacxx.test.cmake.check_source_files
  # gersemi: off
  COMMAND
    "${CMAKE_COMMAND}"
      -D "LIBCUDACXX_SOURCE_DIR=${libcudacxx_SOURCE_DIR}"
      -P "${CMAKE_CURRENT_LIST_DIR}/check_source_files.cmake"
  # gersemi: on
)
