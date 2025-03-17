# Bring in CMAKE_INSTALL_* vars
include(GNUInstallDirs)

# CCCL has no installable binaries, no need to build before installing:
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

# Usage:
# cccl_generate_install_rules(PROJECT_NAME DEFAULT_ENABLE
#                             [NO_HEADERS]
#                             [HEADER_SUBDIR <subdir1> [subdir2 ...]]
#                             [HEADERS_INCLUDE <pattern1> [pattern2 ...]]
#                             [HEADERS_EXCLUDE <pattern1> [pattern2 ...]]
#                             [PACKAGE]
# )
#
# Options:
# PROJECT_NAME: The case-sensitive name of the project. Used to generate the option flag.
# DEFAULT_ENABLE: Whether the install rules should be enabled by default.
# NO_HEADERS: If set, no install rules will be generated for headers.
# HEADERS_SUBDIRS: If set, a separate install rule will be generated for each subdirectory relative to the project dir.
#                  If not set, <CCCL_SOURCE_DIR>/<PROJECT_NAME_LOWER>/<PROJECT_NAME_LOWER> will be used.
# HEADERS_INCLUDE: A list of globbing patterns that match installable header files.
# HEADERS_EXCLUDE: A list of globbing patterns that match header files to exclude from installation.
# PACKAGE: If set, install the project's CMake package.
#
# Notes:
# - The generated cache option will be named <PROJECT_NAME>_ENABLE_INSTALL_RULES.
# - The header globs are applied relative to <CCCL_SOURCE_DIR>/<PROJECT_NAME_LOWER>/<SUBDIR>.
# - The cmake package is assumed to be located at <CCCL_SOURCE_DIR>/lib/cmake/<PROJECT_NAME_LOWER>.
# - If a <PROJECT_NAME_LOWER>-header-search.cmake.in file exists in the CMake package directory,
#   it will be configured and installed.
#
function(cccl_generate_install_rules project_name enable_rules_by_default)
  set(options PACKAGE NO_HEADERS)
  set(oneValueArgs)
  set(multiValueArgs HEADERS_SUBDIRS HEADERS_INCLUDE HEADERS_EXCLUDE)
  cmake_parse_arguments(CGIR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  string(TOLOWER ${project_name} project_name_lower)
  set(project_source_dir "${CCCL_SOURCE_DIR}/${project_name_lower}")
  set(header_dest_dir "${CMAKE_INSTALL_INCLUDEDIR}")
  set(package_source_dir "${CCCL_SOURCE_DIR}/lib/cmake/${project_name_lower}")
  set(package_dest_dir "${CMAKE_INSTALL_LIBDIR}/cmake/")
  set(header_search_template "${package_source_dir}/${project_name_lower}-header-search.cmake.in")
  set(header_search_temporary "${CCCL_BINARY_DIR}/${project_name_lower}-header-search.cmake")

  if (NOT DEFINED CGIR_HEADERS_SUBDIRS)
    set(CGIR_HEADERS_SUBDIRS "${project_name_lower}")
  endif()

  set(flag_name ${project_name}_ENABLE_INSTALL_RULES)
  option(${flag_name} "Enable installation of ${project_name} files." ${enable_rules_by_default})
  if (${flag_name})
    # Headers:
    if (NOT CGIR_NO_HEADERS)
      foreach(subdir IN LISTS CGIR_HEADERS_SUBDIRS)
        set(header_globs)
        if(DEFINED CGIR_HEADERS_INCLUDE OR DEFINED CGIR_HEADERS_EXCLUDE)
          set(header_globs "FILES_MATCHING")

          foreach (header_glob IN LISTS CGIR_HEADERS_INCLUDE)
            list(APPEND header_globs "PATTERN" "${header_glob}")
          endforeach()

          foreach (header_glob IN LISTS CGIR_HEADERS_EXCLUDE)
            list(APPEND header_globs "PATTERN" "${header_glob}" "EXCLUDE")
          endforeach()
        endif()

        install(
          DIRECTORY "${project_source_dir}/${subdir}"
          DESTINATION "${header_dest_dir}"
          ${header_globs}
        )
      endforeach()
    endif()

    # CMake package:
    install(
      DIRECTORY "${package_source_dir}"
      DESTINATION "${package_dest_dir}"
      REGEX .*header-search.cmake.* EXCLUDE
    )

    # Header search infra:
    if (EXISTS "${header_search_template}")
      # Need to configure a file to store the infix specified in
      # CMAKE_INSTALL_INCLUDEDIR since it can be defined by the user
      set(_CCCL_RELATIVE_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
      if(_CCCL_RELATIVE_LIBDIR MATCHES "^${CMAKE_INSTALL_PREFIX}")
        # libdir is an abs string that starts with prefix
        string(LENGTH "${CMAKE_INSTALL_PREFIX}" to_remove)
        string(SUBSTRING "${_CCCL_RELATIVE_LIBDIR}" ${to_remove} -1 relative)
        # remove any leading "/""
        string(REGEX REPLACE "^/(.)" "\\1" _CCCL_RELATIVE_LIBDIR "${relative}")
      elseif(_CCCL_RELATIVE_LIBDIR MATCHES "^/")
        message(FATAL_ERROR "CMAKE_INSTALL_LIBDIR ('${CMAKE_INSTALL_LIBDIR}') must be a relative path or an absolute path under CMAKE_INSTALL_PREFIX ('${CMAKE_INSTALL_PREFIX}')")
      endif()
      set(install_location "${_CCCL_RELATIVE_LIBDIR}/cmake/${project_name_lower}")

      # Transform to a list of directories, replace each directory with "../"
      # and convert back to a string
      string(REGEX REPLACE "/" ";" from_install_prefix "${install_location}")
      list(TRANSFORM from_install_prefix REPLACE ".+" "../")
      list(JOIN from_install_prefix "" from_install_prefix)

      configure_file("${header_search_template}" "${header_search_temporary}" @ONLY)
      install(
        FILES "${header_search_temporary}"
        DESTINATION "${install_location}"
      )
    endif()

  endif()
endfunction()

include("${CMAKE_CURRENT_LIST_DIR}/install/cccl.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/install/cub.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/install/cudax.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/install/libcudacxx.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/install/thrust.cmake")
