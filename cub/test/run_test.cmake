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
  message("  - GPU_TEST_RUNNER: Optional.")
  message("    - Path to rapids-cmake's run_gpu_test.cmake.")
  message("    - Used to bind the test to its CTest-allocated GPU when CTest")
  message("      supplies resource groups (GPU resource scheduling).")
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
if (NOT DEFINED TEST)
  usage()
  message(FATAL_ERROR "TEST must be defined")
endif()

if (NOT DEFINED ARGS)
  set(ARGS)
endif()

if (NOT DEFINED TYPE)
  set(TYPE "none")
endif()

if (NOT DEFINED MODE)
  if (DEFINED ENV{CCCL_TEST_MODE})
    message(STATUS "Using CCCL_TEST_MODE from env: $ENV{CCCL_TEST_MODE}")
    set(MODE $ENV{CCCL_TEST_MODE})
  else()
    set(MODE "none")
  endif()
elseif (NOT MODE)
  set(MODE "none")
endif()

if (MODE STREQUAL "none")
  set(command_to_run ${TEST})
  set(command_args ${ARGS})
elseif (MODE MATCHES "^compute-sanitizer-(.*)$")
  set(tool ${CMAKE_MATCH_1})

  # All test cases in these tests take an excessive amount of time to execute with the compute-sanitizer tools.
  # Just skip the whole test.
  if (
    (
      "${TEST}" MATCHES "/cub.+test.*large_offsets.*"
      AND
        (
          tool STREQUAL "initcheck"
          OR tool STREQUAL "racecheck"
          OR tool STREQUAL "synccheck"
        )
    )
    OR
    # These take >2000s on racecheck to complete, even with reduced seeds, skipping large offsets, etc.
    (
      "${TEST}" MATCHES "/cub.+test.device_segmented_(radix_sort|reduce).*"
      AND (tool STREQUAL "racecheck")
    )
  )
    message(
      FATAL_ERROR
      "CCCL_SKIP_TEST:\n${TEST} takes an excessive amount of time to execute with ${tool}."
    )
  endif()

  # These tests intentionally don't launch any kernels or make any CUDA API calls:
  if ("${TEST}" MATCHES "/cub.*.test.cdp_variant_state.*")
    message(
      FATAL_ERROR
      "CCCL_SKIP_TEST:\n${TEST} intentionally doesn't use CUDA APIs. Skipping."
    )
  endif()

  # The CUB debug test intentionally throws CUDA errors to test error handling:
  if ("${TEST}" MATCHES "/cub.*test.debug$")
    message(
      FATAL_ERROR
      "CCCL_SKIP_TEST:\n${TEST} intentionally throws CUDA errors. Skipping."
    )
  endif()

  if (TYPE STREQUAL "Catch2")
    list(APPEND ARGS "--durations" "yes" "~[skip-cs-${tool}]")
  endif()

  # Compile the compute-sanitizer args:
  # gersemi: off
  set(cs_general_args
    "--tool" "${tool}"
    "--suppressions" "${CCCL_SOURCE_DIR}/ci/compute-sanitizer-suppressions.xml"
    "--check-device-heap" "yes"
    "--check-bulk-copy" "yes"
    "--check-exit-code" "yes"
    "--check-warpgroup-mma" "yes"
    "--error-exitcode" "1"
    "--nvtx" "true"
  )
  # gersemi: on
  if (tool STREQUAL "memcheck")
    # gersemi: off
    set(cs_tool_args
      "--leak-check" "full"
      "--padding" "512"
      "--track-stream-ordered-races" "all"
    )
    # gersemi: on
  elseif (tool STREQUAL "racecheck")
    set(cs_tool_args)
  elseif (tool STREQUAL "initcheck")
    set(cs_tool_args)
  elseif (tool STREQUAL "synccheck")
    set(cs_tool_args)
  endif()

  set(command_to_run compute-sanitizer)
  # gersemi: off
  set(command_args
    ${cs_general_args}
    ${cs_tool_args}
    ${TEST} ${ARGS}
  )
  # gersemi: on
else()
  usage()
  message(FATAL_ERROR "Invalid MODE: ${MODE}")
endif()

if (DEFINED ENV{CTEST_RESOURCE_GROUP_COUNT} AND DEFINED GPU_TEST_RUNNER)
  # CTest assigned a GPU resource to this test. Let the rapids-cmake runner
  # set CUDA_VISIBLE_DEVICES before launching the command.
  include("${GPU_TEST_RUNNER}")
else()
  run_command(${command_to_run} ${command_args})
endif()
