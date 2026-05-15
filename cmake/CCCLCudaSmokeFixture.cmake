include_guard(GLOBAL)

# CUDA runtime smoke fixture: centralized FIXTURES_REQUIRED attachment.
#
# Strategy: shadow add_test() with a macro that calls the original (available
# as _add_test after redefinition) and then, when CCCL_CUDA_SMOKE_ENABLED is
# on, attaches FIXTURES_REQUIRED=${CCCL_CUDA_SMOKE_FIXTURE} to the freshly
# registered test. Because this is a macro, it expands at the call site -- the
# set_tests_properties write therefore happens in the same directory scope as
# the underlying add_test(), which is what CMake requires.
#
# This replaces the scattered per-listfile cccl_attach_cuda_smoke_fixture_here()
# calls and the previous patch inside cccl_add_executable(): a single shadow
# covers every test registered via add_test(), wherever it lives.
#
# Filtering:
#   - The smoke target itself is always skipped (never required-by-itself).
#   - An optional CCCL_CUDA_SMOKE_NAME_REGEX CMake variable (set e.g. by
#     thrust/CMakeLists.txt) restricts the match. Stored as a variable rather
#     than a directory property so it propagates through add_subdirectory()
#     automatically. If unset, the macro defaults to
#     ^(cub|thrust|libcudacxx|cudax)\\..*$ so third-party deps such as
#     Catch2 / NVTX / nvbench are not accidentally gated on the CUDA smoke.
#
# Include this file once at root scope BEFORE any add_subdirectory() that may
# register tests; the include_guard(GLOBAL) ensures the macro is installed
# only once.

set(_CCCL_CUDA_SMOKE_DEFAULT_REGEX "^(cub|thrust|libcudacxx|cudax)\\..*$")

macro(add_test)
  _add_test(${ARGV})

  if (CCCL_CUDA_SMOKE_ENABLED)
    # Extract the test name. add_test supports two forms:
    #   add_test(NAME <name> COMMAND ...)
    #   add_test(<name> <command> [args...])    -- legacy
    set(_cccl_smoke_argv ${ARGV})
    list(LENGTH _cccl_smoke_argv _cccl_smoke_argc)
    set(_cccl_smoke_name "")
    if (_cccl_smoke_argc GREATER 0)
      list(GET _cccl_smoke_argv 0 _cccl_smoke_first)
      if ("${_cccl_smoke_first}" STREQUAL "NAME" AND _cccl_smoke_argc GREATER 1)
        list(GET _cccl_smoke_argv 1 _cccl_smoke_name)
      else()
        set(_cccl_smoke_name "${_cccl_smoke_first}")
      endif()
    endif()

    if (
      NOT "${_cccl_smoke_name}" STREQUAL ""
      AND NOT "${_cccl_smoke_name}" STREQUAL "${CCCL_CUDA_SMOKE_TARGET}"
    )
      if (
        DEFINED CCCL_CUDA_SMOKE_NAME_REGEX
        AND NOT "${CCCL_CUDA_SMOKE_NAME_REGEX}" STREQUAL ""
      )
        set(_cccl_smoke_regex "${CCCL_CUDA_SMOKE_NAME_REGEX}")
      else()
        set(_cccl_smoke_regex "${_CCCL_CUDA_SMOKE_DEFAULT_REGEX}")
      endif()
      if (_cccl_smoke_name MATCHES "${_cccl_smoke_regex}")
        set_tests_properties(
          "${_cccl_smoke_name}"
          PROPERTIES FIXTURES_REQUIRED "${CCCL_CUDA_SMOKE_FIXTURE}"
        )
      endif()
      unset(_cccl_smoke_regex)
    endif()

    unset(_cccl_smoke_argv)
    unset(_cccl_smoke_argc)
    unset(_cccl_smoke_first)
    unset(_cccl_smoke_name)
  endif()
endmacro()
