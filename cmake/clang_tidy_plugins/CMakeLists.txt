cmake_minimum_required(VERSION 3.26.4)

list(APPEND CMAKE_MESSAGE_CONTEXT "clang_tidy_plugins")

# LLVM requires C to be enabled
project(clang_tidy_plugins LANGUAGES C CXX)

if (PROJECT_IS_TOP_LEVEL)
  message(
    FATAL_ERROR
    "clang-tidy plugins must be built as part of the main CCCL build"
  )
endif()

# We need compile-commands in all cases so that clang-tidy tests work
if (NOT CMAKE_EXPORT_COMPILE_COMMANDS)
  message(FATAL_ERROR "Clang-tidy tests require compile_commands.json")
endif()

# Some package managers (conda) have separate LLVM and clang development packages.
# Finding just LLVM or just clang does not guarantee that the other is installed, so need
# to check both exist upfront here.
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG HINTS "${LLVM_CMAKE_DIR}/..")

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(CCCL_LLVM_LIBS core)

# This is because the clang-tidy job in CI is usually run when CCCL itself is in a debug
# build (since this allows clang-tidy to see asserts in the code). But we want to build
# the actual plugins in release mode so they aren't slow.
option(
  CCCL_CLANG_TIDY_PLUGIN_RESPECT_CMAKE_BUILD_TYPE
  "Respect the current cmake build type when building clang-tidy plugins instead of always building in release mode"
  OFF
)
mark_as_advanced(CCCL_CLANG_TIDY_PLUGIN_RESPECT_CMAKE_BUILD_TYPE)

find_program(
  CCCL_FILECHECK
  NAMES FileCheck FileCheck-${LLVM_VERSION_MAJOR}
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

#[=======================================================================[.rst:
cccl_create_clang_tidy_plugin
-----------------------------

Add a check source to the existing ``cccl.tidy.plugins`` module.

Arguments
^^^^^^^^^

``SOURCE``
  One source file, with an absolute path or a path relative to
  ``CMAKE_CURRENT_SOURCE_DIR``.

#]=======================================================================]
function(cccl_create_clang_tidy_plugin SOURCE)
  list(APPEND CMAKE_MESSAGE_CONTEXT "create_clang_tidy_plugin")

  target_sources(cccl.tidy.plugins PRIVATE "${SOURCE}")
endfunction()

function(_cccl_parse_clang_tidy_config source_file out_var)
  # Since the clang-tidy config is parsed at configure-time we need to rerun cmake if the
  # file changes.
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${source_file}"
  )

  set(
    config
    [==[{
      "Checks": ["-*", "-clang-diagnostic-*"],
      "CheckOptions": {},
      "ExtraArgs": [],
      "HeaderFileExtensions": ["", "h", "hh", "hpp", "hxx", "cuh", "inl", "ipp"],
      "SystemHeaders": false
    }]==]
  )

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
function(_cccl_add_build_fixture target)
  add_test(
    NAME "${target}.build"
    COMMAND
      "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --config "$<CONFIG>"
      --target "${target}"
  )
  set_tests_properties(
    "${target}.build"
    PROPERTIES FIXTURES_SETUP "${target}.built" RESOURCE_LOCK cccl_build
  )
endfunction()

function(_cccl_test_requires_target test_name target)
  set_property(
    TEST "${test_name}"
    APPEND
    PROPERTY FIXTURES_REQUIRED "${target}.built"
  )
endfunction()

#[=======================================================================[.rst:
cccl_create_clang_tidy_test
--------------------------

Create a test for the particular clang-tidy plugin.

Arguments
^^^^^^^^^

``plugin``
  The check group used in the test name, such as ``modernize_type_traits``.

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
  target_compile_options(
    "${target}.compile"
    PRIVATE -Wno-pragma-clang-attribute
  )

  _cccl_parse_clang_tidy_config("${source}" config)
  add_test(
    NAME "${target}"
    COMMAND
      "${CMAKE_COMMAND}" "-DSOURCE=${source}"
      "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/${test_name}.out"
      "-DPLUGIN=$<TARGET_FILE:cccl.tidy.plugins>" "-DCONFIG=${config}"
      "-DBUILD_DIR=${CMAKE_BINARY_DIR}" "-DCLANG_TIDY=${CCCL_CLANG_TIDY}"
      "-DFILECHECK=${CCCL_FILECHECK}" -P
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/run_filecheck.cmake"
  )
  _cccl_test_requires_target("${target}" cccl.tidy.plugins)
endfunction()

block()
  add_library(cccl.tidy.plugins MODULE)

  target_include_directories(
    cccl.tidy.plugins
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CLANG_INCLUDE_DIRS}
      ${LLVM_INCLUDE_DIRS}
  )
  target_compile_definitions(cccl.tidy.plugins PRIVATE ${LLVM_DEFINITIONS_LIST})
  target_link_libraries(
    cccl.tidy.plugins
    PRIVATE cccl::clang-tidy ${CCCL_LLVM_LIBS} clangToolingRefactoring
  )

  if (NOT CCCL_CLANG_TIDY_PLUGIN_RESPECT_CMAKE_BUILD_TYPE)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      message(
        STATUS
        "Building clang-tidy plugin with release flags, even for debug builds"
      )
    endif()
    target_compile_options(cccl.tidy.plugins PRIVATE -O3 -g)
  endif()
  cccl_ensure_metatargets(cccl.tidy.plugins)

  _cccl_add_build_fixture(cccl.tidy.plugins)
endblock()

add_subdirectory(avoid_internal_linkage_in_headers)
add_subdirectory(modernize_type_traits)
add_subdirectory(prefer_cuda_traits)
add_subdirectory(simplify_type_traits)
