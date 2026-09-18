include_guard(GLOBAL)

#[=======================================================================[.rst:
_cccl_create_tidy_plugins
-------------------------

Provide the plugin targets that :command:`cccl_tidy_init` needs to configure
``clang-tidy`` with CCCL checks.

Arguments
^^^^^^^^^

``ret_var``
  The variable in which to store the plugin target list in the parent scope.

#]=======================================================================]
function(_cccl_create_tidy_plugins ret_var)
  list(APPEND CMAKE_MESSAGE_CONTEXT "create_tidy_plugins")

  # Use an absolute path for binary directory because we are not certain where this
  # function will be called from the first time. Also, because we need to pass the path to
  # the plugins directly to clang-tidy, we need a stable directory where they will be
  # output.
  add_subdirectory(
    "${CCCL_SOURCE_DIR}/cmake/clang_tidy_plugins"
    "${CCCL_BINARY_DIR}/cccl_clang_tidy_plugins"
  )

  get_property(plugins GLOBAL PROPERTY CCCL_TIDY_PLUGINS)
  if (NOT plugins)
    # TODO (jfaibussowit):
    #
    # Enable this check once we actually have clang-tidy plugins
    #
    # message(
    #   FATAL_ERROR
    #   "clang-tidy plugins failed to propagate the list of configured plugins."
    # )
  endif()
  set(${ret_var} "${plugins}" PARENT_SCOPE)
endfunction()

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

  _cccl_create_tidy_plugins(all_plugins)

  set(load_cmds)
  foreach (plugin IN LISTS all_plugins)
    list(APPEND load_cmds "--load='$<TARGET_FILE:${plugin}>'")
  endforeach()
  list(JOIN load_cmds " " CCCL_CLANG_TIDY_PLUGINS)

  # configure_file alone does not support generator expressions (which are needed for the
  # clang-tidy plugins), while file(GENERATE) does not support @VAR@ substitutions. So we
  # need to do it ourselves
  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/run_clang_tidy.sh.in"
    "${CCCL_RUN_CLANG_TIDY_SCRIPT}.tmp"
    @ONLY
  )
  file(
    GENERATE OUTPUT "${CCCL_RUN_CLANG_TIDY_SCRIPT}"
    INPUT "${CCCL_RUN_CLANG_TIDY_SCRIPT}.tmp"
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
