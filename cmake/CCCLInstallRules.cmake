# Bring in CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)

# CCCL has no installable binaries, no need to build before installing:
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

install(DIRECTORY "${CCCL_SOURCE_DIR}/lib/cmake/cccl"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/"
)
