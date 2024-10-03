# Check source code for issues that can be found by pattern matching:
cccl_add_ctest(thrust.test.cmake.check_source_files
  LABEL_NAME
  COMMAND
    "${CMAKE_COMMAND}"
      -D "Thrust_SOURCE_DIR=${Thrust_SOURCE_DIR}"
      -P "${CMAKE_CURRENT_LIST_DIR}/check_source_files.cmake"
)
