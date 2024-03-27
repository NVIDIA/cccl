# Bring in CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)

# Thrust is a header library; no need to build anything before installing:
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

install(DIRECTORY "${Thrust_SOURCE_DIR}/thrust"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.inl"
)

install(DIRECTORY "${Thrust_SOURCE_DIR}/thrust/cmake/"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/thrust"
  REGEX .*header-search.cmake.* EXCLUDE
)

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
set(install_location "${_CCCL_RELATIVE_LIBDIR}/cmake/thrust")

# Transform to a list of directories, replace each directory with "../"
# and convert back to a string
string(REGEX REPLACE "/" ";" from_install_prefix "${install_location}")
list(TRANSFORM from_install_prefix REPLACE ".+" "../")
list(JOIN from_install_prefix "" from_install_prefix)

configure_file("${Thrust_SOURCE_DIR}/thrust/cmake/thrust-header-search.cmake.in"
  "${Thrust_BINARY_DIR}/thrust/cmake/thrust-header-search.cmake"
  @ONLY)
install(FILES "${Thrust_BINARY_DIR}/thrust/cmake/thrust-header-search.cmake"
  DESTINATION "${install_location}")
