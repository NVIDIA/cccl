find_package(Python3 REQUIRED COMPONENTS Interpreter)

file(GLOB output_files "${CMAKE_CURRENT_SOURCE_DIR}/*.output")
foreach (output_file IN LISTS output_files)
  get_filename_component(test_name "${output_file}" NAME_WE)
  set(dirty_file "${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.dirty_files")
  if (NOT EXISTS "${dirty_file}")
    message(FATAL_ERROR "Missing dirty file for ${test_name}")
  endif()

  add_test(
    NAME cccl.ci.test.inspect_changes.${test_name}
    # gersemi: off
    COMMAND
      "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/run_inspect_changes_test.py"
        "--python" "${Python3_EXECUTABLE}"
        "--script" "${PROJECT_SOURCE_DIR}/ci/inspect_changes.py"
        "--dirty" "${dirty_file}"
        "--expected" "${output_file}"
    # gersemi: on
  )
endforeach()
