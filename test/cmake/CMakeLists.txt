set(cmake_opts
  -D "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  -D "CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
  -D "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
)

# Temporary installation prefix for tests against installed project:
set(tmp_install_prefix "${CMAKE_CURRENT_BINARY_DIR}/test_install")

foreach (root_type IN ITEMS SOURCE INSTALL)
  if (root_type STREQUAL "INSTALL")
    set(cccl_root "${tmp_install_prefix}")
  else()
    set(cccl_root "${CCCL_SOURCE_DIR}")
  endif()

  set(comps DEFAULT Thrust CUB libcudacxx)
  if (CCCL_ENABLE_UNSTABLE)
    list(APPEND comps cudax)
  endif()

  foreach (components IN LISTS comps)
    set(package_types CCCL)
    if (NOT components STREQUAL "DEFAULT")
      list(APPEND package_types NATIVE)
    endif()
    if (root_type STREQUAL "SOURCE")
      list(APPEND package_types SUBDIR)
    endif()
    foreach (package_type IN LISTS package_types)
      string(TOLOWER "${root_type}.${package_type}.${components}" suffix)
      cccl_add_compile_test(test_name
        cccl.test.cmake
        test_export
        "${suffix}"
        ${cmake_opts}
        -D "CCCL_ROOT=${cccl_root}"
        -D "COMPONENTS=${components}"
        -D "PACKAGE_TYPE=${package_type}"
        -D "CCCL_ENABLE_UNSTABLE=${CCCL_ENABLE_UNSTABLE}"
      )

      if (root_type STREQUAL "INSTALL")
        set_tests_properties(${test_name} PROPERTIES FIXTURES_REQUIRED install_tree)
      endif()
    endforeach() # package_type
  endforeach() # components
endforeach() # root_type

################################################################################
# Install tree fixtures
add_test(NAME cccl.test.cmake.install_tree.install
  COMMAND "${CMAKE_COMMAND}"
    --install "${CCCL_BINARY_DIR}"
    --prefix "${tmp_install_prefix}"
)
set_tests_properties(cccl.test.cmake.install_tree.install PROPERTIES
  FIXTURES_SETUP install_tree
)

add_test(NAME cccl.test.cmake.install_tree.cleanup
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${tmp_install_prefix}"
)
set_tests_properties(cccl.test.cmake.install_tree.cleanup PROPERTIES
  FIXTURES_CLEANUP install_tree
)
