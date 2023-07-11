set(cmake_opts
  -D "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  -D "CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
  -D "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
)

# Temporary installation prefix for tests against installed project:
set(tmp_install_prefix "${CMAKE_CURRENT_BINARY_DIR}/test_install")

# Add a build-and-test CTest.
# - full_test_name_var will be set to the full name of the test.
# - subdir is the relative path to the test project directory.
# - test_id is used to generate a unique name for this test, allowing the
#   subdir to be reused.
# - Any additional args will be passed to the project configure step.
function(cccl_add_compile_test full_test_name_var subdir test_id)
  set(test_name cccl.test.cmake.${subdir}.${test_id})
  set(src_dir "${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${subdir}/${test_id}")
  add_test(NAME ${test_name}
    COMMAND "${CMAKE_CTEST_COMMAND}"
      --build-and-test "${src_dir}" "${build_dir}"
      --build-generator "${CMAKE_GENERATOR}"
      --build-options
        ${cmake_opts}
        ${ARGN}
      --test-command "${CMAKE_CTEST_COMMAND}" --output-on-failure
  )
  set(${full_test_name_var} ${test_name} PARENT_SCOPE)
endfunction()

foreach (root_type IN ITEMS SOURCE INSTALL)
  if (root_type STREQUAL "INSTALL")
    set(cccl_root "${tmp_install_prefix}")
  else()
    set(cccl_root "${CCCL_SOURCE_DIR}")
  endif()

  foreach (components IN ITEMS DEFAULT Thrust CUB libcudacxx)
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
        test_export
        "${suffix}"
        -D "CCCL_ROOT=${cccl_root}"
        -D "ROOT_TYPE=${root_type}"
        -D "COMPONENTS=${components}"
        -D "PACKAGE_TYPE=${package_type}"
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
