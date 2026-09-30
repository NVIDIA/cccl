execute_process(
  COMMAND
    "${CLANG_TIDY}" "--load=${PLUGIN}" "--config=${CONFIG}"
    "--export-fixes=${OUTPUT}.yaml" -p "${BUILD_DIR}" "${SOURCE}"
  OUTPUT_FILE "${OUTPUT}"
  ERROR_FILE "${OUTPUT}"
  COMMAND_ECHO STDOUT
  RESULT_VARIABLE result
)

if (NOT "${result}" STREQUAL "0")
  file(READ "${OUTPUT}" output)
  message(FATAL_ERROR "clang-tidy failed (${result}):\n${output}")
endif()

execute_process(
  COMMAND
    "${FILECHECK}" "${SOURCE}" --check-prefix=CHECK-FIXES
    "--input-file=${OUTPUT}.yaml"
  COMMAND_ECHO STDOUT
  COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
  COMMAND
    "${FILECHECK}" "${SOURCE}" --check-prefix=CHECK-MESSAGES
    "--input-file=${OUTPUT}"
  COMMAND_ECHO STDOUT
  COMMAND_ERROR_IS_FATAL ANY
)
