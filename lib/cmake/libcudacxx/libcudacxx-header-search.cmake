# Parse version information from version header:
set(_libcudacxx_VERSION_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../libcudacxx/include")
if(EXISTS "${_libcudacxx_VERSION_INCLUDE_DIR}/cuda/std/detail/__config")
  set(_libcudacxx_VERSION_INCLUDE_DIR "${_libcudacxx_VERSION_INCLUDE_DIR}" CACHE FILEPATH "" FORCE) # Clear old result
  set_property(CACHE _libcudacxx_VERSION_INCLUDE_DIR PROPERTY TYPE INTERNAL)
endif()
