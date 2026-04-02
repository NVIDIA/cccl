include_guard(GLOBAL)

#[=======================================================================[.rst:
cccl_tidy_init
--------------

Initialize ``clang-tidy`` support and define the global ``cccl.tidy`` target. It must be
called before adding any CCCL ``clang-tidy`` targets.

Subsequent calls to this functions are no-ops.

Result Variables
^^^^^^^^^^^^^^^^

  ``CCCL_TIDY_INITIALIZED`` set to true in the parent scope.

#]=======================================================================]
function(cccl_tidy_init)
  list(APPEND CMAKE_MESSAGE_CONTEXT "tidy_init")

  if (CCCL_TIDY_INITIALIZED)
    return()
  endif()

  find_program(CCCL_CLANG_TIDY clang-tidy REQUIRED)

  add_custom_target(cccl.tidy COMMENT "clang-tidy CCCL")

  # Do not set to cache; multiple separate instances of CCCL in a build should not
  # conflict.
  set(CCCL_TIDY_INITIALIZED TRUE)
  set(CCCL_TIDY_INITIALIZED TRUE PARENT_SCOPE)
endfunction()

#[=======================================================================[.rst:
cccl_tidy_make_subproject_target
--------------------------------

Create a meta target per sub-project that depends on all the targets for that
subproject. It itself will depend on the ``cccl.tidy target``. For example, this will
create:

- cub.tidy
- libcudacxx.tidy
- thrust.tidy

etc. This allows running clang-tidy over just a subset of the repository.

The generated target name depends on the current value of ``PROJECT_NAME``.

Arguments
^^^^^^^^^

``result_var``
  The variable in which to store the created target name.

#]=======================================================================]
function(cccl_tidy_make_subproject_target result_var)
  list(APPEND CMAKE_MESSAGE_CONTEXT "tidy_make_subproject_target")

  if (NOT CCCL_TIDY_INITIALIZED)
    # For the cccl.tidy target
    message(FATAL_ERROR "Must call cccl_tidy_init() first")
  endif()

  string(TOLOWER "${PROJECT_NAME}.tidy" target_name)

  if (NOT TARGET "${target_name}")
    add_custom_target("${target_name}" COMMENT "clang-tidy ${PROJECT_NAME}")
    add_dependencies(cccl.tidy "${target_name}")
  endif()
  set(${result_var} "${target_name}" PARENT_SCOPE)
endfunction()

#[=======================================================================[.rst:
cccl_tidy_add_target
--------------------

Create per-source ``clang-tidy`` targets and attach them to both the global ``cccl.tidy``
target and per sub-project target (e.g. ``cub.tidy``)

.. note::

  :command:`cccl_tidy_init` must be called before using this function to establish the
  global ``cccl.tidy`` target.

If ``CCCL_ENABLE_CLANG_TIDY`` is false, this does nothing (except error-check the function
call signature).

Passing the same source file multiple times is allowed. A target is created for it only
once.

If ``SOURCES`` is empty, this function does nothing.

Arguments
^^^^^^^^^

``SOURCES``
  List of source files to analyze. Paths may be absolute or relative. Relative paths are
  resolved against ``CMAKE_CURRENT_SOURCE_DIR``.

#]=======================================================================]
function(cccl_tidy_add_target)
  list(APPEND CMAKE_MESSAGE_CONTEXT "tidy_add_target")

  set(options)
  set(one_value_args)
  set(multi_value_args SOURCES)

  cmake_parse_arguments(
    _cccl
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  if (_cccl_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unrecognized arguments: ${_cccl_UNPARSED_ARGUMENTS}")
  endif()

  # It is still possible to call this function even if clang-tidy has not been
  # disabled. We handle this gracefully to avoid complicating the callsite.
  #
  # This must come before the CCCL_TIDY_INITIALIZED check because that is only called when
  # CCCL_ENABLE_CLANG_TIDY is true.
  if (NOT CCCL_ENABLE_CLANG_TIDY)
    return()
  endif()

  if (NOT CCCL_TIDY_INITIALIZED)
    message(FATAL_ERROR "Must call cccl_tidy_init() first")
  endif()

  cccl_tidy_make_subproject_target(subproject_target)

  foreach (src IN LISTS _cccl_SOURCES)
    cmake_path(SET src NORMALIZE "${src}")
    if (NOT IS_ABSOLUTE "${src}")
      cmake_path(SET src NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    endif()

    cmake_path(
      RELATIVE_PATH src
      BASE_DIRECTORY "${CCCL_SOURCE_DIR}"
      OUTPUT_VARIABLE rel_src
    )
    string(MAKE_C_IDENTIFIER "${rel_src}" tidy_target)
    set(tidy_target "${tidy_target}.tidy")

    if (TARGET "${tidy_target}")
      # We have seen this file before
      return()
    endif()

    add_custom_target(
      "${tidy_target}"
      DEPENDS "${src}"
      COMMAND
        ${CCCL_CLANG_TIDY} #
        --use-color #
        --quiet #
        --extra-arg=-Wno-error=unused-command-line-argument #
        --extra-arg=-D_CCCL_CLANG_TIDY_INVOKED=1 #
        -p "${CMAKE_BINARY_DIR}" #
        "${src}"
      COMMENT "clang-tidy ${rel_src}"
    )

    add_dependencies("${subproject_target}" "${tidy_target}")
  endforeach()
endfunction()
