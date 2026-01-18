# project_name: The name of the project when calling `find_package`. Case sensitive.
# `PACKAGE_FILEBASE` the name of the project in the config files, ie. ${PACKAGE_FILEBASE}-config.cmake.
# `PACKAGE_PATH` the absolute path to the project's CMake package config files.
function(cccl_add_subdir_helper project_name)
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
    CCCL_SUBDIR
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  if (NOT DEFINED CCCL_SUBDIR_PACKAGE_FILEBASE)
    string(TOLOWER "${project_name}" CCCL_SUBDIR_PACKAGE_FILEBASE)
  endif()

  if (NOT DEFINED CCCL_SUBDIR_PACKAGE_PATH)
    set(
      CCCL_SUBDIR_PACKAGE_PATH
      "${CCCL_SOURCE_DIR}/lib/cmake/${CCCL_SUBDIR_PACKAGE_FILEBASE}"
    )
  endif()

  set(
    package_prefix
    "${CCCL_SUBDIR_PACKAGE_PATH}/${CCCL_SUBDIR_PACKAGE_FILEBASE}"
  )

  set(CMAKE_FIND_PACKAGE_NAME ${project_name})
  set(${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS)
  if (DEFINED CCCL_SUBDIR_REQUIRED_COMPONENTS)
    list(
      APPEND ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS
      ${CCCL_SUBDIR_REQUIRED_COMPONENTS}
    )
    foreach (component IN LISTS CCCL_SUBDIR_REQUIRED_COMPONENTS)
      set(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED_${component} TRUE)
    endforeach()
  endif()

  if (DEFINED CCCL_SUBDIR_OPTIONAL_COMPONENTS)
    list(
      APPEND ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS
      ${CCCL_SUBDIR_OPTIONAL_COMPONENTS}
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
      "${CCCL_SUBDIR_PACKAGE_PATH}"
      CACHE PATH
      "Path to ${project_name} package"
    )
  endif()
endfunction()
