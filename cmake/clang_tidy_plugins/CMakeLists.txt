cmake_minimum_required(VERSION 3.26.4)

list(APPEND CMAKE_MESSAGE_CONTEXT "clang_tidy_plugins")

# LLVM requires C to be enabled
project(clang_tidy_plugins LANGUAGES C CXX)

option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile commands" ON)

if (PROJECT_IS_TOP_LEVEL)
  message(
    FATAL_ERROR
    "clang-tidy plugins must be built as part of the main CCCL build"
  )
endif()

# Some package managers (conda) have separate LLVM and clang development packages.
# Finding just LLVM or just clang does not guarantee that the other is installed, so need
# to check both exist upfront here.
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG HINTS "${LLVM_CMAKE_DIR}/..")

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(CCCL_LLVM_LIBS core)

define_property(GLOBAL PROPERTY CCCL_TIDY_PLUGINS)

#[=======================================================================[.rst:
cccl_create_clang_tidy_plugin
-----------------------------

Define a plugin that extends ``clang-tidy`` with CCCL checks.

Use this function to make custom checks available to the CCCL ``clang-tidy`` targets.

Arguments
^^^^^^^^^

``TARGET``
  The name of the plugin target to create.

``SOURCES``
  List of source files for the plugin.

#]=======================================================================]
function(cccl_create_clang_tidy_plugin TARGET)
  list(APPEND CMAKE_MESSAGE_CONTEXT "create_clang_tidy_plugin")

  set(options)
  set(one_value_args)
  set(multi_value_keywords SOURCES)
  cmake_parse_arguments(
    _CCCL
    "${options}"
    "${one_value_args}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  add_library("${TARGET}" MODULE "${_CCCL_SOURCES}")

  target_include_directories(
    "${TARGET}"
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CLANG_INCLUDE_DIRS}
      ${LLVM_INCLUDE_DIRS}
  )
  target_compile_definitions("${TARGET}" PRIVATE ${LLVM_DEFINITIONS_LIST})
  target_link_libraries(
    "${TARGET}"
    PRIVATE ${CCCL_LLVM_LIBS} clangTidy clangToolingRefactoring
  )

  set_property(GLOBAL APPEND PROPERTY CCCL_TIDY_PLUGINS "${TARGET}")
endfunction()

# Reuse CCCLAddTidyTarget's executable, or find it for standalone builds.
find_program(
  CCCL_FILECHECK
  NAMES
    FileCheck
    FileCheck-23
    FileCheck-22
    FileCheck-21
    FileCheck-20
    FileCheck-19
    FileCheck-18
  HINTS "${LLVM_TOOLS_BINARY_DIR}"
  REQUIRED
)

execute_process(
  COMMAND ${CCCL_FILECHECK} --version
  OUTPUT_VARIABLE version
  ERROR_VARIABLE version
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

message(STATUS "Found FileCheck: ${CCCL_FILECHECK} (${version})")

function(_cccl_parse_clang_tidy_config source_file out_var)
  # Since the clang-tidy config is parsed at configure-time we need to rerun cmake if the
  # file changes.
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${source_file}"
  )

  set(config [==[{"Checks":["-*"],"CheckOptions":{},"ExtraArgs":[]}]==])

  file(STRINGS "${source_file}" lines)
  set(in_config FALSE)
  set(found_checks FALSE)

  foreach (line IN LISTS lines)
    if (NOT in_config)
      if (line MATCHES "^[ \t]*//[ \t]*CLANG_TIDY_CONFIG_BEGIN[ \t]*$")
        set(in_config TRUE)
      endif()
      continue()
    endif()
    if (line MATCHES "^[ \t]*//[ \t]*CLANG_TIDY_CONFIG_END[ \t]*$")
      set(in_config FALSE)
      break()
    endif()

    if (line MATCHES "^[ \t]*//[ \t]*Checks[ \t]*:[ \t]*(.*)$")
      set(found_checks TRUE)
      string(REPLACE "," ";" enabled_checks "${CMAKE_MATCH_1}")
      list(TRANSFORM enabled_checks STRIP)
      foreach (check IN LISTS enabled_checks)
        string(JSON index LENGTH "${config}" Checks)
        string(JSON config SET "${config}" Checks ${index} "\"${check}\"")
      endforeach()
    elseif (
      line MATCHES "^[ \t]*//[ \t]*CheckOptions[ \t]*:[ \t]*([^=]+)=(.*)$"
    )
      string(STRIP "${CMAKE_MATCH_1}" option_name)
      string(STRIP "${CMAKE_MATCH_2}" option_value)
      string(
        JSON config
        SET "${config}"
        CheckOptions
        "${option_name}"
        "\"${option_value}\""
      )
    elseif (line MATCHES "^[ \t]*//[ \t]*ExtraArgs[ \t]*:[ \t]*(.+)$")
      string(STRIP "${CMAKE_MATCH_1}" argument)
      string(JSON index LENGTH "${config}" ExtraArgs)
      string(JSON config SET "${config}" ExtraArgs ${index} "\"${argument}\"")
    else()
      message(
        FATAL_ERROR
        "${source_file}: unrecognized clang-tidy config line: ${line}"
      )
    endif()
  endforeach()

  if (in_config)
    message(FATAL_ERROR "${source_file}: missing CLANG_TIDY_CONFIG_END")
  endif()
  if (NOT found_checks)
    message(FATAL_ERROR "${source_file}: missing Checks: directive")
  endif()

  set(${out_var} "${config}" PARENT_SCOPE)
endfunction()

# We have to do this wonky test fixture setup business because CMake does not allow you to
# specify target dependencies between tests, because tests are not actual real cmake
# targets. So you need to create a separate "test" which actually builds the thing you
# want and set that as the required fixture for your other tests.
function(_cccl_ensure_plugin_is_built_for_test test_target plugin)
  set(plugin_build_target "${plugin}.build")
  set(fixture "${plugin}.build_fixture")

  if (NOT TEST ${plugin_build_target})
    add_test(
      NAME "${plugin_build_target}"
      COMMAND
        "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --config "$<CONFIG>"
        --target "${plugin}"
    )
    set_tests_properties(
      "${plugin_build_target}"
      PROPERTIES FIXTURES_SETUP "${fixture}" RESOURCE_LOCK cccl_build
    )
  endif()
  set_tests_properties(
    "${test_target}"
    PROPERTIES FIXTURES_REQUIRED "${fixture}"
  )
endfunction()

#[=======================================================================[.rst:
cccl_create_clang_tidy_test
--------------------------

Register a test for a clang-tidy plugin's diagnostics and suggested fixes.

Arguments
^^^^^^^^^

``plugin``
  The existing plugin target that provides the checks under test.

``source``
  The test source file, with an absolute path or a path relative to the current
  source directory. Place these directives between ``// CLANG_TIDY_CONFIG_BEGIN``
  and ``// CLANG_TIDY_CONFIG_END``. Unrecognized lines inside the block are errors.

  * ``// Checks: check-name[,check-name...]`` selects the checks to enable.
    At least one line is required. Multiple lines add checks.
  * ``// CheckOptions: key=value`` sets an optional check option.
    Multiple lines set multiple options. The last value for a repeated key wins.
  * ``// ExtraArgs: argument`` appends one compiler argument.
    Multiple lines append multiple arguments.
  * ``CHECK-MESSAGES`` FileCheck directives specify expected diagnostics.
  * ``CHECK-FIXES`` FileCheck directives specify expected replacements in the
    exported fixes YAML.

#]=======================================================================]
function(cccl_create_clang_tidy_test plugin source)
  list(APPEND CMAKE_MESSAGE_CONTEXT "create_clang_tidy_test")

  cmake_path(ABSOLUTE_PATH source NORMALIZE)
  cmake_path(GET source STEM test_name)
  set(target "cccl.test.tidy.${plugin}.${test_name}")

  # We don't actually care about compiling this (and in fact, we never want to!). The
  # point of adding this here is to get the corresponding entries into the compilation
  # database so that clang-tidy knows how to interpret the file properly.
  add_library("${target}.compile" OBJECT EXCLUDE_FROM_ALL "${source}")
  target_link_libraries(
    "${target}.compile"
    PRIVATE cccl.compiler_interface libcudacxx::libcudacxx
  )

  _cccl_parse_clang_tidy_config("${source}" config)
  add_test(
    NAME "${target}"
    COMMAND
      "${CMAKE_COMMAND}" "-DSOURCE=${source}"
      "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/${test_name}.out"
      "-DPLUGIN=$<TARGET_FILE:${plugin}>" "-DCONFIG=${config}"
      "-DBUILD_DIR=${CMAKE_BINARY_DIR}" "-DCLANG_TIDY=${CCCL_CLANG_TIDY}"
      "-DFILECHECK=${CCCL_FILECHECK}" -P
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/run_filecheck.cmake"
  )
  _cccl_ensure_plugin_is_built_for_test("${target}" "${plugin}")
endfunction()

# The parent enables testing after configuring this directory.
enable_testing()

add_subdirectory(prefer_cuda_traits)
