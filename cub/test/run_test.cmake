#
# Launch a test, optionally enabling runtime sanitizers, etc.
#

function(usage)
  message("Usage:")
  message("  cmake -D CCCL_SOURCE_DIR=/path/to/cccl \\")
  message("        -D TEST=bin/test.exe \\")
  message("        -D ARGS=\"arg1;arg2;arg3\" \\")
  message("        -D TYPE=Catch2 \\")
  message("        -D MODE=compute-sanitizer-memcheck \\")
  message("        -P cccl/cub/test/run_test.cmake")
  message("")
  message("  - CCCL_SOURCE_DIR: Required.  Path to the CCCL source directory.")
  message("  - TEST: Required.  Path to the test executable.")
  message("  - ARGS: Optional.  Arguments to pass to the test executable.")
  message("  - TYPE: Optional.")
  message("    - The test framework used by the test executable.")
  message("    - Must be one of the following:")
  message("    -  \"none\" (default)")
  message("    -  \"Catch2\"")
  message("  - MODE: Optional.")
  message("    - May be set through CCCL_TEST_MODE env var.")
  message("    - Must be one of the following:")
  message("    -  \"none\" (default)")
  message("    -  \"compute-sanitizer-memcheck\"")
  message("    -  \"compute-sanitizer-racecheck\"")
  message("    -  \"compute-sanitizer-initcheck\"")
  message("    -  \"compute-sanitizer-synccheck\"")
endfunction()

# Usage:
#   run_command(COMMAND [ARGS...])
#
# The command is printed before it is executed.
# The new process's stdout, sterr are redirected to the current process.
# The current process will exit with an error if the new process exits with a non-zero status.
function(run_command command)
  set(command_str "${command}")
  list(APPEND command_str ${ARGN})
  list(JOIN command_str " " command_str)
  message(STATUS ">> Running:\n\t${command_str}")
  execute_process(COMMAND ${command} ${ARGN} RESULT_VARIABLE result)
  if (NOT result EQUAL 0)
    message(FATAL_ERROR ">> Exit Status: ${result}")
  else()
    message(STATUS ">> Exit Status: ${result}")
  endif()
endfunction()

######################################################################

# Parse arguments
if(NOT DEFINED TEST)
  usage()
  message(FATAL_ERROR "TEST must be defined")
endif()

if(NOT DEFINED ARGS)
  set(ARGS)
endif()

if(NOT DEFINED TYPE)
  set(TYPE "none")
endif()

if(NOT DEFINED MODE)
  if(DEFINED ENV{CCCL_TEST_MODE})
    message(STATUS "Using CCCL_TEST_MODE from env: $ENV{CCCL_TEST_MODE}")
    set(MODE $ENV{CCCL_TEST_MODE})
  else()
    set(MODE "none")
  endif()
elseif(NOT MODE)
  set(MODE "none")
endif()

if (MODE STREQUAL "none")
  run_command(${TEST} ${ARGS})
elseif (MODE MATCHES "^compute-sanitizer-(.*)$")
  set(tool ${CMAKE_MATCH_1})
  if (TYPE STREQUAL "Catch2")
    list(APPEND ARGS
      "--durations" "yes"
      "~[skip-cs-${tool}]"
    )
  endif()

  # All test cases in these tests take an excessive amount of time to execute with the compute-sanitizer tools.
  # Just skip the whole test.
  # The CUB segmented kernels in particular seem to run into issues here.
  if (("${TEST}" MATCHES "/cub.+test.device_segmented_radix_sort" AND tool STREQUAL "racecheck") OR
      ("${TEST}" MATCHES "/cub.+test.device_segmented_reduce" AND tool STREQUAL "initcheck") OR
      ("${TEST}" MATCHES "/cub.+test.device_scan_large_offsets" AND (tool STREQUAL "initcheck" OR tool STREQUAL "racecheck")))
    message(FATAL_ERROR "CCCL_SKIP_TEST:\n${TEST} takes an excessive amount of time to execute with ${tool}.")
  endif()

  # The CUB debug test intentionally throws CUDA errors to test error handling:
  if ("${TEST}" MATCHES "/cub.*test.debug$")
    message(FATAL_ERROR "CCCL_SKIP_TEST:\n${TEST} intentionally throws CUDA errors. Skipping.")
  endif()

  run_command(compute-sanitizer
    --tool ${tool}
    --suppressions "${CCCL_SOURCE_DIR}/ci/compute-sanitizer-suppressions.xml"
    # TODO Figure out what the min version needed is for this:
    # --check-bulk-copy yes
    --check-device-heap yes
    --leak-check full
    --padding 512
    --track-stream-ordered-races all
    --check-warpgroup-mma yes
    --require-cuda-init no # Disable "no CUDA APIs used" failures
    --check-exit-code yes
    --error-exitcode 1
    --nvtx true
    ${TEST} ${ARGS}
  )
else()
  usage()
  message(FATAL_ERROR "Invalid MODE: ${MODE}")
endif()
