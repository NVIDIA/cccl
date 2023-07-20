# RunExample.cmake
# Inputs:
#
# Variable           | Type     | Description
# -------------------|----------|------------------------------------------------
# EXAMPLE_EXECUTABLE | FilePath | Path to the example executable

execute_process(
  COMMAND "${EXAMPLE_EXECUTABLE}"
  RESULT_VARIABLE exit_code
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

if (NOT exit_code EQUAL 0)
  message(FATAL_ERROR "${EXAMPLE_EXECUTABLE} failed (${exit_code}):\n${stderr}")
else()
  message(STATUS "${EXAMPLE_EXECUTABLE} succeeded:\n${stdout}")
endif()
