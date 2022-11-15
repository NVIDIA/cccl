option(libcudacxx_ENABLE_INSTALL_RULES
  "Enable installation of libcudacxx" ${LIBCUDACXX_TOPLEVEL_PROJECT}
)

if (NOT libcudacxx_ENABLE_INSTALL_RULES)
  return()
endif()

# Bring in CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)

# Libcudacxx headers
install(DIRECTORY "${libcudacxx_SOURCE_DIR}/include/cuda"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERN CMakeLists.txt EXCLUDE
)
install(DIRECTORY "${libcudacxx_SOURCE_DIR}/include/nv"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERN CMakeLists.txt EXCLUDE
)

# Libcudacxx cmake package
install(DIRECTORY "${libcudacxx_SOURCE_DIR}/lib/cmake/libcudacxx"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
  PATTERN *.cmake.in EXCLUDE
)

# Need to configure a file to store CMAKE_INSTALL_INCLUDEDIR
# since it can be defined by the user. This is common to work around collisions
# with the CTK installed headers.
configure_file("${libcudacxx_SOURCE_DIR}/lib/cmake/libcudacxx/libcudacxx-header-search.cmake.in"
  "${libcudacxx_BINARY_DIR}/lib/cmake/libcudacxx/libcudacxx-header-search.cmake"
  @ONLY
)
install(FILES "${libcudacxx_BINARY_DIR}/lib/cmake/libcudacxx/libcudacxx-header-search.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/libcudacxx"
)
