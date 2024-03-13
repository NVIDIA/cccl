# Bring in CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)

# Headers
install(DIRECTORY "${CudaNext_SOURCE_DIR}/include/cuda"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERN CMakeLists.txt EXCLUDE
)

# CMake package
install(DIRECTORY "${CudaNext_SOURCE_DIR}/lib/cmake/CudaNext"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
)
