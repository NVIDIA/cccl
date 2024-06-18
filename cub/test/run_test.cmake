#
# Launch a test, optionally enabling runtime sanitizers, etc.
#

function(usage)
  message("Usage:")
  message("  cmake -D TEST=bin/test.exe \\")
  message("        -D ARGS=\"arg1 arg2\" \\")
  message("        -D MODE=compute-sanitizer-memcheck \\")
  message("        -P cccl/cub/test/run_test.cmake")
  message("")
  message("  - TEST: Required.  Path to the test executable.")
  message("  - ARGS: Optional.  Arguments to pass to the test executable.")
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
  list(APPEND command ${ARGN})
  list(JOIN command " " command_str)
  message(STATUS ">> Running:\n\t${command_str}")
  execute_process(COMMAND ${command} RESULT_VARIABLE result)
  if (NOT result EQUAL 0)
    message(FATAL_ERROR ">> Exit Status: ${result}")
  else()
    message(STATUS ">>Exit Status: ${result}")
  endif()
endfunction()

######################################################################

# Parse arguments
if(NOT DEFINED TEST)
  usage()
  message(FATAL_ERROR "TEST must be defined")
endif()

if(NOT DEFINED ARGS)
  set(ARGS "")
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
  run_command(compute-sanitizer
    --tool ${tool}
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
