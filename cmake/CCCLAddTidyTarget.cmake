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

  execute_process(
    COMMAND ${CCCL_CLANG_TIDY} --version
    OUTPUT_VARIABLE version
    ERROR_VARIABLE version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )

  message(STATUS "Found clang-tidy: ${CCCL_CLANG_TIDY} (${version})")

  add_custom_target(cccl.tidy COMMENT "clang-tidy CCCL")

  set(
    CCCL_RUN_CLANG_TIDY_SCRIPT
    "${CMAKE_CURRENT_BINARY_DIR}/run_clang_tidy.sh"
  )
  set(CCCL_RUN_CLANG_TIDY_SCRIPT "${CCCL_RUN_CLANG_TIDY_SCRIPT}" PARENT_SCOPE)

  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/run_clang_tidy.sh.in"
    "${CCCL_RUN_CLANG_TIDY_SCRIPT}"
    @ONLY
  )
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
      continue()
    endif()

    add_custom_target(
      "${tidy_target}"
      DEPENDS "${src}" "${CCCL_RUN_CLANG_TIDY_SCRIPT}"
      COMMAND ${CCCL_RUN_CLANG_TIDY_SCRIPT} "${src}"
      COMMENT "clang-tidy ${rel_src}"
    )

    add_dependencies("${subproject_target}" "${tidy_target}")
  endforeach()
endfunction()

#[=======================================================================[.rst:
cccl_tidy_add_header_sweep
--------------------------

Run a restricted set of ``clang-tidy`` checks over a set of headers, analyzing each header
as host C++ in a translation unit of its own.

.. note::

  :command:`cccl_tidy_init` must be called before using this function to establish the
  global ``cccl.tidy`` target.

Unlike :command:`cccl_tidy_add_target`, this does not analyze translation units taken from
the build's compilation database. It generates a one-line translation unit per header and
analyzes it with an explicit host C++ command line, which has two consequences: the sweep
covers headers that no translation unit happens to include, and it works for headers that
clang's CUDA front end cannot compile.

Creates one target per header, a sweep target named ``<subproject>.tidy.<label>`` that
depends on all of them, and attaches the sweep target to the per-subproject target (e.g.
``cudax.tidy``), hence to the global ``cccl.tidy`` target.

If ``CCCL_ENABLE_CLANG_TIDY`` is false, this does nothing (except error-check the function
call signature).

Arguments
^^^^^^^^^

``label``
  Names the sweep. Used for target names and progress messages.

``project_include_path``
  The path to the project's include directory, relative to ``CCCL_SOURCE_DIR``. Globs and
  the generated ``#include`` directives are resolved against it.

``CHECKS``
  The value of ``clang-tidy``'s ``--checks`` option, e.g. ``-*,bugprone-exception-escape``.

``GLOBS``
  All headers matching these globbing patterns are analyzed, unless they also match
  ``EXCLUDES``.

``EXCLUDES``
  Headers matching these globbing patterns are not analyzed.

``INCLUDE_DIRECTORIES``
  Directories to pass as ``-I``.

``SYSTEM_INCLUDE_DIRECTORIES``
  Directories to pass as ``-isystem``.

#]=======================================================================]
function(cccl_tidy_add_header_sweep label project_include_path)
  list(APPEND CMAKE_MESSAGE_CONTEXT "tidy_add_header_sweep")

  set(options)
  set(one_value_args CHECKS)
  set(
    multi_value_args
    GLOBS
    EXCLUDES
    INCLUDE_DIRECTORIES
    SYSTEM_INCLUDE_DIRECTORIES
  )

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

  if (NOT _cccl_CHECKS)
    message(FATAL_ERROR "CHECKS is required")
  endif()

  # See the corresponding comment in cccl_tidy_add_target.
  if (NOT CCCL_ENABLE_CLANG_TIDY)
    return()
  endif()

  if (NOT CCCL_TIDY_INITIALIZED)
    message(FATAL_ERROR "Must call cccl_tidy_init() first")
  endif()

  cccl_tidy_make_subproject_target(subproject_target)

  set(base_path "${CCCL_SOURCE_DIR}/${project_include_path}")

  set(globs)
  foreach (glob IN LISTS _cccl_GLOBS)
    list(APPEND globs "${base_path}/${glob}")
  endforeach()

  file(GLOB_RECURSE headers RELATIVE "${base_path}" CONFIGURE_DEPENDS ${globs})

  if (_cccl_EXCLUDES)
    set(excludes)
    foreach (exclude IN LISTS _cccl_EXCLUDES)
      list(APPEND excludes "${base_path}/${exclude}")
    endforeach()

    file(
      GLOB_RECURSE header_excludes
      RELATIVE "${base_path}"
      CONFIGURE_DEPENDS
      ${excludes}
    )

    if (header_excludes)
      list(REMOVE_ITEM headers ${header_excludes})
    endif()
  endif()

  if (NOT headers)
    message(FATAL_ERROR "No headers matched the given GLOBS")
  endif()

  # CCCL headers declare themselves system headers unless _CCCL_NO_SYSTEM_HEADER is
  # defined. clang-tidy honors that declaration and drops every diagnostic raised in such a
  # header, so without this definition the sweep silently finds nothing.
  set(flags "-std=c++${CMAKE_CXX_STANDARD}" -D_CCCL_NO_SYSTEM_HEADER)
  foreach (dir IN LISTS _cccl_INCLUDE_DIRECTORIES)
    list(APPEND flags -I "${dir}")
  endforeach()
  foreach (dir IN LISTS _cccl_SYSTEM_INCLUDE_DIRECTORIES)
    list(APPEND flags -isystem "${dir}")
  endforeach()

  set(sweep_target "${subproject_target}.${label}")
  add_custom_target(
    "${sweep_target}"
    COMMENT "clang-tidy ${label} ${PROJECT_NAME}"
  )
  add_dependencies("${subproject_target}" "${sweep_target}")

  foreach (header IN LISTS headers)
    string(MAKE_C_IDENTIFIER "${header}" header_id)
    set(tu "${CMAKE_CURRENT_BINARY_DIR}/tidy/${label}/${header_id}.cpp")
    file(CONFIGURE OUTPUT "${tu}" CONTENT "#include <${header}>\n")

    set(tidy_target "${sweep_target}.${header_id}")
    add_custom_target(
      "${tidy_target}"
      DEPENDS "${base_path}/${header}" "${CCCL_RUN_CLANG_TIDY_SCRIPT}"
      # Arguments after `--` replace the compilation database entry for the translation
      # unit, which has none: it is generated for this sweep alone.
      COMMAND
        ${CCCL_RUN_CLANG_TIDY_SCRIPT} "--checks=${_cccl_CHECKS}" "${tu}" --
        ${flags}
      COMMENT "clang-tidy ${label} ${header}"
    )

    add_dependencies("${sweep_target}" "${tidy_target}")
  endforeach()
endfunction()
