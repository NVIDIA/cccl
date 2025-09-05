find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
  NAME cccl.ci.test.parse_error
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
