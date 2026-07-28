set(LIBCUDACXX_SUPPORTED_DEBUGGERS lldb gdb)
set(LIBCUDACXX_DEBUGGING_BASE_TARGET libcudacxx.test.debugging)

function(libcudacxx_init_debugger_testing enabled_var)
  # Unconditionally create the umbrella target to make CMakePresets easier to setup. If we
  # don't have the debuggers available, this target doesn't do anything
  add_custom_target(${LIBCUDACXX_DEBUGGING_BASE_TARGET})

  # Windows lldb is broken sometimes (see
  # https://github.com/llvm/llvm-project/issues/74073) so merely finding it does not mean
  # it is functional. In any case, it is good to validate the binary since even a found
  # lldb/gdb on other platforms that ends up not working is annoying to work around
  function(validator result_var item)
    execute_process(
      COMMAND ${item} --version
      OUTPUT_QUIET
      ERROR_QUIET
      RESULT_VARIABLE result
      TIMEOUT 10
    )

    if (result EQUAL 0)
      set(${result_var} TRUE PARENT_SCOPE)
    else()
      set(${result_var} FALSE PARENT_SCOPE)
    endif()
  endfunction()

  set(enabled FALSE)
  foreach (debugger IN LISTS LIBCUDACXX_SUPPORTED_DEBUGGERS)
    string(TOUPPER "${debugger}" DEBUGGER_UPPER)

    find_program(
      LIBCUDACXX_${DEBUGGER_UPPER}
      NAMES "${debugger}"
      VALIDATOR validator
    )

    if (LIBCUDACXX_${DEBUGGER_UPPER})
      set(debugger_exe "${LIBCUDACXX_${DEBUGGER_UPPER}}")
      message(STATUS "Found ${debugger}: ${debugger_exe}")
      execute_process(
        COMMAND ${debugger_exe} --version
        OUTPUT_VARIABLE version
        ERROR_VARIABLE version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
      )
      message(STATUS "${debugger_exe} --version: ${version}")
    endif()

    set(default OFF)
    if (LIBCUDACXX_${DEBUGGER_UPPER})
      set(default ON)
    endif()

    option(
      LIBCUDACXX_ENABLE_${DEBUGGER_UPPER}_TESTING
      "Run libcudacxx pretty-printer tests with ${debugger}"
      ${default}
    )

    if (
      LIBCUDACXX_ENABLE_${DEBUGGER_UPPER}_TESTING
      AND NOT LIBCUDACXX_${DEBUGGER_UPPER}
    )
      message(
        FATAL_ERROR
        "LIBCUDACXX_ENABLE_${DEBUGGER_UPPER}_TESTING is enabled, but ${debugger} was not found"
      )
    endif()

    set(
      LIBCUDACXX_${DEBUGGER_UPPER}
      "${LIBCUDACXX_${DEBUGGER_UPPER}}"
      PARENT_SCOPE
    )
    set(
      LIBCUDACXX_ENABLE_${DEBUGGER_UPPER}_TESTING
      "${LIBCUDACXX_ENABLE_${DEBUGGER_UPPER}_TESTING}"
      PARENT_SCOPE
    )

    if (LIBCUDACXX_ENABLE_${DEBUGGER_UPPER}_TESTING)
      set(enabled TRUE)
    endif()
  endforeach()

  set(${enabled_var} ${enabled} PARENT_SCOPE)
endfunction()

libcudacxx_init_debugger_testing(testing_enabled)

if (NOT testing_enabled)
  return()
endif()

#[=======================================================================[.rst:
libcudacxx_add_pretty_printer_test
---------------------------------

Build a CUDA pretty-printer scenario and register its enabled debugger tests.

The function creates the executable target
``libcudacxx.test.debugging.<NAME>`` with host debug information and optimization
disabled. For each enabled debugger, it registers a serial CTest named
``libcudacxx.test.debugging.<debugger>.<NAME>``. The test uses the debugger's
formatter entry point and the ``<debugger>.expected`` file in the caller's source
directory.

Arguments
^^^^^^^^^

``NAME``
  Scenario name used in the executable target, CTest names, and diagnostic output.

``SOURCES``
  Source files used to build the CUDA scenario executable. Relative paths are resolved
  against the caller's source directory.

``CASES``
  Ordered runner arguments describing the debugger stops and expressions. Each case
  has the form ``--case <breakpoint> <caller-frame-index> <section-name>
  <expression>``.

#]=======================================================================]
function(libcudacxx_add_pretty_printer_test)
  set(options)
  set(one_value_arguments NAME)
  set(multi_value_arguments SOURCES CASES)
  cmake_parse_arguments(
    pretty_printer
    "${options}"
    "${one_value_arguments}"
    "${multi_value_arguments}"
    ${ARGN}
  )

  if (pretty_printer_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR
      "Unrecognized arguments: ${pretty_printer_UNPARSED_ARGUMENTS}"
    )
  endif()
  if (NOT pretty_printer_NAME)
    message(FATAL_ERROR "libcudacxx_add_pretty_printer_test requires NAME")
  endif()
  if (NOT pretty_printer_SOURCES)
    message(FATAL_ERROR "libcudacxx_add_pretty_printer_test requires SOURCES")
  endif()
  if (NOT pretty_printer_CASES)
    message(FATAL_ERROR "libcudacxx_add_pretty_printer_test requires CASES")
  endif()

  set(target_name "${LIBCUDACXX_DEBUGGING_BASE_TARGET}.${pretty_printer_NAME}")
  cccl_add_executable(
    ${target_name}
    DIALECT 17
    NO_CLANG_TIDY
    SOURCES ${pretty_printer_SOURCES}
  )
  target_compile_options(
    ${target_name}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g> $<$<COMPILE_LANGUAGE:CUDA>:-O0>
  )
  target_link_libraries(${target_name} PRIVATE libcudacxx.compiler_interface)

  foreach (debugger IN LISTS LIBCUDACXX_SUPPORTED_DEBUGGERS)
    string(TOUPPER "${debugger}" DEBUGGER_UPPER)

    if (NOT LIBCUDACXX_ENABLE_${DEBUGGER_UPPER}_TESTING)
      continue()
    endif()

    set(executable "${LIBCUDACXX_${DEBUGGER_UPPER}}")
    set(
      formatter
      "${libcudacxx_SOURCE_DIR}/share/libcudacxx/${debugger}/__init__.py"
    )
    set(test_name "${target_name}.${debugger}")
    add_test(
      NAME ${test_name}
      COMMAND
        # gersemi: off
      "${Python_EXECUTABLE}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/run_pretty_printer_test.py"
      --debugger "${debugger}"
      --debugger-executable "${executable}"
      --program "$<TARGET_FILE:${target_name}>"
      --formatter-init "${formatter}"
      --expected "${CMAKE_CURRENT_SOURCE_DIR}/${debugger}.expected"
      --output-log "${CMAKE_CURRENT_BINARY_DIR}/${debugger}.log"
      ${pretty_printer_CASES}
      # gersemi: on
    )
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 60)
  endforeach()
endfunction()

add_subdirectory(buffer)
add_subdirectory(memory_resource)
