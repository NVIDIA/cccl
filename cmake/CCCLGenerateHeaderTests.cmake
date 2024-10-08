# Usage:
# cccl_generate_header_tests(<target_name> <project_include_path>
#                            [cccl_configure_target options]
#                            [LANGUAGE <CXX|CUDA>]
#                            [HEADER_TEMPLATE <template>]
#                            [GLOBS <glob1> [glob2 ...]]
#                            [EXCLUDES <glob1> [glob2 ...]]
#                            [HEADERS <header1> [header2 ...]]
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
#
# Notes:
# - The header globs are applied relative to <project_include_path>.
# - If no HEADER_TEMPLATE is provided, a default template will be used.
# - The HEADER_TEMPLATE will be configured for each header, with the following variables:
#   - @header@: The path to the target header, relative to <project_include_path>.
function(cccl_generate_header_tests target_name project_include_path)
  set(options)
  set(oneValueArgs LANGUAGE HEADER_TEMPLATE)
  set(multiValueArgs GLOBS EXCLUDES HEADERS DEFINES)
  cmake_parse_arguments(CGHT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Setup defaults
  if (NOT DEFINED CGHT_LANGUAGE)
    set(CGHT_LANGUAGE CUDA)
  endif()

  if (NOT DEFINED CGHT_HEADER_TEMPLATE)
    set(CGHT_HEADER_TEMPLATE "${CCCL_SOURCE_DIR}/cmake/header_test.cu.in")
  endif()

  # Derived vars:
  if (${CGHT_LANGUAGE} STREQUAL "CXX")
    set(extension "cpp")
  elseif(${CGHT_LANGUAGE} STREQUAL "CUDA")
    set(extension "cu")
  else()
    message(FATAL_ERROR "Unsupported language: ${CGHT_LANGUAGE}")
  endif()

  set(cccl_configure_target_options ${CGHT_UNPARSED_ARGUMENTS})
  set(base_path "${CCCL_SOURCE_DIR}/${project_include_path}")

  # Prepend the basepath to all globbing expressions:
  if (DEFINED CGHT_GLOBS)
    set(globs)
    foreach (glob IN LISTS CGHT_GLOBS)
      list(APPEND globs "${base_path}/${glob}")
    endforeach()
    set(CGHT_GLOBS ${globs})
  endif()
  if (DEFINED CGHT_EXCLUDES)
    set(excludes)
    foreach (exclude IN LISTS CGHT_EXCLUDES)
      list(APPEND excludes "${base_path}/${exclude}")
    endforeach()
    set(CGHT_EXCLUDES ${excludes})
  endif()

  # Determine header list
  set(headers)

  # Add globs:
  if (DEFINED CGHT_GLOBS)
    file(GLOB_RECURSE headers
      RELATIVE "${base_path}"
      CONFIGURE_DEPENDS
      ${CGHT_GLOBS}
    )
  endif()

  # Remove excludes:
  if (DEFINED CGHT_EXCLUDES)
    file(GLOB_RECURSE header_excludes
      RELATIVE "${base_path}"
      CONFIGURE_DEPENDS
      ${CGHT_EXCLUDES}
    )
    list(REMOVE_ITEM headers ${header_excludes})
  endif()

  # Add explicit headers:
  if (DEFINED CGHT_HEADERS)
    list(APPEND headers ${CGHT_HEADERS})
  endif()

  # Cleanup:
  list(REMOVE_DUPLICATES headers)

  # Configure header templates:
  set(header_srcs)
  foreach (header IN LISTS headers)
    set(header_src "${CMAKE_CURRENT_BINARY_DIR}/headers/${target_name}/${header}.${extension}")
    configure_file(${CGHT_HEADER_TEMPLATE} ${header_src} @ONLY)
    list(APPEND header_srcs ${header_src})
  endforeach()

  # Object library that compiles each header:
  add_library(${target_name} OBJECT ${header_srcs})
  cccl_configure_target(${target_name} ${cccl_configure_target_options})

endfunction()
