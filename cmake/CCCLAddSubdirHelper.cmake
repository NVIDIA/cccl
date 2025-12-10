# project_name: The name of the project when calling `find_package`. Case sensitive.
# `PACKAGE_FILEBASE` the name of the project in the config files,
#   ie. ${PACKAGE_FILEBASE}-config.cmake.
#   Defaults to the lower case version of `project_name`.
# `PACKAGE_PATH` the absolute path to the project's CMake package config files.
#   Defaults to "${CCCL_SOURCE_DIR}/lib/cmake/${PACKAGE_FILEBASE}".
# `REQUIRED_COMPONENTS` list of components to mark as required when finding the package.
# `OPTIONAL_COMPONENTS` list of components to mark as optional when finding the package.
#
function(cccl_add_subdir_helper project_name)
  string(TOLOWER "${project_name}" project_name_lower)
  set(options)
  set(
    oneValueArgs
    PACKAGE_PATH
    PACKAGE_FILEBASE
    REQUIRED_COMPONENTS
    OPTIONAL_COMPONENTS
  )
  set(multiValueArgs)
  cmake_parse_arguments(
    self
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )
  cccl_parse_arguments_error_checks(
    "cccl_add_subdir_helper"
    ERROR_UNPARSED
    DEFAULT_VALUES #
      PACKAGE_FILEBASE ${project_name_lower}
      PACKAGE_PATH "${CCCL_SOURCE_DIR}/lib/cmake/${project_name_lower}"
  )

  set(package_prefix "${self_PACKAGE_PATH}/${self_PACKAGE_FILEBASE}")

  set(CMAKE_FIND_PACKAGE_NAME ${project_name})
  set(${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS)
  if (DEFINED self_REQUIRED_COMPONENTS)
    list(
      APPEND ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS
      ${self_REQUIRED_COMPONENTS}
    )
    foreach (component IN LISTS self_REQUIRED_COMPONENTS)
      set(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED_${component} TRUE)
    endforeach()
  endif()

  if (DEFINED self_OPTIONAL_COMPONENTS)
    list(
      APPEND ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS
      ${self_OPTIONAL_COMPONENTS}
    )
  endif()

  # This effectively does a `find_package` actually going through the find_package
  # machinery. Using `find_package` works for the first configure, but creates
  # inconsistencies during subsequent configurations when using CPM..
  #
  # More details are in the discussion at
  # https://github.com/NVIDIA/libcudacxx/pull/242#discussion_r794003857
  include("${package_prefix}-config-version.cmake")
  include("${package_prefix}-config.cmake")

  if (${project_name}_FOUND)
    # Set the dir var so that later `find_package` calls work as expected.
    set(
      ${project_name}_DIR
      "${self_PACKAGE_PATH}"
      CACHE PATH
      "Path to ${project_name} package"
    )
  endif()
endfunction()
