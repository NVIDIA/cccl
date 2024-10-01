# project_name: The name of the project when calling `find_package`. Case sensitive.
# `PACKAGE_FILEBASE` the name of the project in the config files, ie. ${PACKAGE_FILEBASE}-config.cmake.
# `PACKAGE_PATH` the path to the project's CMake package config files.
function(cccl_add_subdir_helper project_name)
  set(options)
  set(oneValueArgs PACKAGE_PATH PACKAGE_FILEBASE REQUIRED_COMPONENTS OPTIONAL_COMPONENTS)
  set(multiValueArgs)
  cmake_parse_arguments(CCCL_SUBDIR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (DEFINED CCCL_SUBDIR_PACKAGE_FILEBASE)
    set(package_filebase "${CCCL_SUBDIR_PACKAGE_FILEBASE}")
  else()
    string(TOLOWER "${project_name}" package_filebase)
  endif()

  if (DEFINED CCCL_SUBDIR_PACKAGE_PATH)
    set(package_prefix "${CCCL_SUBDIR_PACKAGE_PATH}/${package_filebase}")
  else()
    set(package_prefix "${CCCL_SOURCE_DIR}/lib/cmake/${package_filebase}/${package_filebase}")
  endif()

  set(CMAKE_FIND_PACKAGE_NAME ${project_name})
  set(${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS)
  if (DEFINED CCCL_SUBDIR_REQUIRED_COMPONENTS)
    list(APPEND ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS ${CCCL_SUBDIR_REQUIRED_COMPONENTS})
    foreach(component IN LISTS CCCL_SUBDIR_REQUIRED_COMPONENTS)
      set(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED_${component} TRUE)
    endforeach()
  endif()

  if (DEFINED CCCL_SUBDIR_OPTIONAL_COMPONENTS)
    list(APPEND ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS ${CCCL_SUBDIR_OPTIONAL_COMPONENTS})
  endif()

  # This effectively does a `find_package` actually going through the find_package
  # machinery. Using `find_package` works for the first configure, but creates
  # inconsistencies during subsequent configurations when using CPM..
  #
  # More details are in the discussion at
  # https://github.com/NVIDIA/libcudacxx/pull/242#discussion_r794003857
  include("${package_prefix}-config-version.cmake")
  include("${package_prefix}-config.cmake")
endfunction()
