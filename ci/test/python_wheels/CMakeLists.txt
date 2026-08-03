find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
  NAME cccl.ci.test.python_wheels
  COMMAND
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_wheel_scripts.py"
    --collector
    "${PROJECT_SOURCE_DIR}/ci/util/python/collect_wheel_artifacts.py"
    --validator
    "${PROJECT_SOURCE_DIR}/ci/util/python/validate_cccl_wheel_set.py"
)
