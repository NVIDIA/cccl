# Bring in CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)

# CUB is a header library; no need to build anything before installing:
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

install(DIRECTORY "${CUB_SOURCE_DIR}/cub"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
    PATTERN "*.cuh"
)

install(DIRECTORY "${CUB_SOURCE_DIR}/cub/cmake/"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cub"
  REGEX .*header-search.cmake.* EXCLUDE
)
# Need to configure a file to store the infix specified in
# CMAKE_INSTALL_INCLUDEDIR since it can be defined by the user
set(install_location "${CMAKE_INSTALL_LIBDIR}/cmake/cub")
configure_file("${CUB_SOURCE_DIR}/cub/cmake/cub-header-search.cmake.in"
  "${CUB_BINARY_DIR}/cub/cmake/cub-header-search.cmake"
  @ONLY)
install(FILES "${CUB_BINARY_DIR}/cub/cmake/cub-header-search.cmake"
  DESTINATION "${install_location}")
