# Usage:
# cccl_generate_header_tests(<target_name> <project_include_path>
#                            [cccl_configure_target options]
#                            [LANGUAGE <CXX|CUDA>]
#                            [HEADER_TEMPLATE <template>]
#                            [GLOBS <glob1> [glob2 ...]]
#                            [EXCLUDES <glob1> [glob2 ...]]
#                            [HEADERS <header1> [header2 ...]]
#                            [PER_HEADER_DEFINES
#                               DEFINE <definition> <regex> [<regex> ...]
#                              [DEFINE <definition> <regex> [<regex> ...]] ...]
#                            [NO_METATARGETS]
#                            [NO_LINK_CHECK]
# )
#
# Options:
# target_name: The name of the meta-target that will build this set of header tests.
# project_include_path: The path to the project's include directory, relative to <CCCL_SOURCE_DIR>.
# cccl_configure_target options: Options to pass to cccl_configure_target. Must appear before any other named arguments.
# LANGUAGE: The language to use for the header tests. Defaults to CUDA.
# HEADER_TEMPLATE: A file that will be used as a template for each header test. The template will be configured for each header.
# GLOBS: All files that match these globbing patterns will be included in the header tests, unless they also match EXCLUDES.
# EXCLUDES: Files that match these globbing patterns will be excluded from the header tests.
# HEADERS: An explicit list of headers to include in the header tests.
# PER_HEADER_DEFINES: A list of definitions to add to specific headers. Each definition is followed by one or more regexes that match the headers it should be applied to.
# NO_METATARGETS: If specified, metatargets will not be created for the header test targets.
# NO_LINK_CHECK: If specified, the link check target will not be created.
#
# Notes:
# - The header globs are applied relative to <project_include_path>.
# - If no HEADER_TEMPLATE is provided, a default template will be used.
# - The HEADER_TEMPLATE will be configured for each header, with the following variables:
#   - @header@: The path to the target header, relative to <project_include_path>.
function(cccl_generate_header_tests target_name project_include_path)
  set(options NO_METATARGETS NO_LINK_CHECK)
  set(oneValueArgs LANGUAGE HEADER_TEMPLATE)
  set(multiValueArgs GLOBS EXCLUDES HEADERS PER_HEADER_DEFINES)
  cmake_parse_arguments(
    self
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )
  cccl_parse_arguments_error_checks(
    "cccl_generate_header_tests"
    DEFAULT_VALUES #
      LANGUAGE "CUDA"
      HEADER_TEMPLATE "${CCCL_SOURCE_DIR}/cmake/header_test.cu.in"
  )

  # Derived vars:
  if (${self_LANGUAGE} STREQUAL "C")
    set(extension "c")
  elseif (${self_LANGUAGE} STREQUAL "CXX")
    set(extension "cpp")
  elseif (${self_LANGUAGE} STREQUAL "CUDA")
    set(extension "cu")
  else()
    message(FATAL_ERROR "Unsupported language: ${self_LANGUAGE}")
  endif()

  set(cccl_configure_target_options ${self_UNPARSED_ARGUMENTS})
  set(base_path "${CCCL_SOURCE_DIR}/${project_include_path}")

  # Prepend the basepath to all globbing expressions:
  if (DEFINED self_GLOBS)
    set(globs)
    foreach (glob IN LISTS self_GLOBS)
      list(APPEND globs "${base_path}/${glob}")
    endforeach()
    set(self_GLOBS ${globs})
  endif()
  if (DEFINED self_EXCLUDES)
    set(excludes)
    foreach (exclude IN LISTS self_EXCLUDES)
      list(APPEND excludes "${base_path}/${exclude}")
    endforeach()
    set(self_EXCLUDES ${excludes})
  endif()

  # Determine header list
  set(headers)

  # Add globs:
  if (DEFINED self_GLOBS)
    file(
      GLOB_RECURSE headers
      RELATIVE "${base_path}"
      CONFIGURE_DEPENDS
      ${self_GLOBS}
    )
  endif()

  # Remove excludes:
  if (DEFINED self_EXCLUDES)
    file(
      GLOB_RECURSE header_excludes
      RELATIVE "${base_path}"
      CONFIGURE_DEPENDS
      ${self_EXCLUDES}
    )
    list(REMOVE_ITEM headers ${header_excludes})
  endif()

  # Add explicit headers:
  if (DEFINED self_HEADERS)
    list(APPEND headers ${self_HEADERS})
  endif()

  # Cleanup:
  list(REMOVE_DUPLICATES headers)

  # Helper function for applying per-header defines:
  # header: The original header filepath
  # src: The generated source file for the header test
  function(cght_apply_per_header_defines header src)
    if (NOT DEFINED self_PER_HEADER_DEFINES)
      return()
    endif()
    set(current_definition)
    foreach (item IN LISTS self_PER_HEADER_DEFINES)
      if (item STREQUAL "DEFINE")
        # New definition
        set(current_definition)
      elseif (NOT current_definition)
        # First item after DEFINE is the definition
        set(current_definition "${item}")
      else()
        # Subsequent items are regexes to match against the header
        if (header MATCHES ${item})
          set_property(
            SOURCE "${src}"
            APPEND
            PROPERTY COMPILE_DEFINITIONS "${current_definition}"
          )
        endif()
      endif()
    endforeach()
  endfunction()

  # Configure header templates:
  set(header_srcs)
  foreach (header IN LISTS headers)
    set(
      header_src
      "${CMAKE_CURRENT_BINARY_DIR}/headers/${target_name}/${header}.${extension}"
    )
    configure_file("${self_HEADER_TEMPLATE}" "${header_src}" @ONLY)
    cght_apply_per_header_defines("${header}" "${header_src}")
    list(APPEND header_srcs "${header_src}")
  endforeach()

  # Object library that compiles each header:
  add_library(${target_name} OBJECT ${header_srcs})
  cccl_configure_target(${target_name} ${cccl_configure_target_options})
  if (NOT self_NO_METATARGETS)
    cccl_ensure_metatargets(${target_name})
  endif()

  if (NOT self_NO_LINK_CHECK)
    # Check that all functions in headers are either template functions or inline:
    set(link_target ${target_name}.link_check)
    cccl_add_executable(
      ${link_target}
      SOURCES "${CCCL_SOURCE_DIR}/cmake/link_check_main.${extension}"
      NO_METATARGETS
    )
    # Linking both ${target_name} and $<TARGET_OBJECTS:${target_name}> forces CMake to
    # link the same objects twice. The compiler will complain about duplicate symbols if
    # any functions are missing inline markup.
    target_link_libraries(
      ${link_target}
      PRIVATE #
        ${target_name}
        $<TARGET_OBJECTS:${target_name}>
    )
  endif()
endfunction()
