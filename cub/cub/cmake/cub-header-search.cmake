# Parse version information from version.h in source tree
set(_CUB_VERSION_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
if(EXISTS "${_CUB_VERSION_INCLUDE_DIR}/cub/version.cuh")
  set(_CUB_VERSION_INCLUDE_DIR "${_CUB_VERSION_INCLUDE_DIR}" CACHE FILEPATH "" FORCE) # Clear old result
  set_property(CACHE _CUB_VERSION_INCLUDE_DIR PROPERTY TYPE INTERNAL)
endif()
