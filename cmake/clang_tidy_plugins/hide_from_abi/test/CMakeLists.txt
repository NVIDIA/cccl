list(APPEND CMAKE_MESSAGE_CONTEXT "test")

file(GLOB test_srcs CONFIGURE_DEPENDS "*.hpp")
file(GLOB cu_test_srcs CONFIGURE_DEPENDS "*.cuh")

function(cccl_add_abi_tests LANG)
  foreach (src IN LISTS ARGN)
    cmake_path(GET src STEM name)
    foreach (
      extension
      IN
      ITEMS "" h hh hpp hxx cuh inl ipp
    )
      set(header "${CMAKE_CURRENT_BINARY_DIR}/${name}_${extension}")
      if (NOT extension STREQUAL "")
        string(APPEND header ".${extension}")
      endif()
      configure_file("${src}" "${header}" COPYONLY)
      # Parse headers directly so the diagnostic line filter includes their declarations.
      set_source_files_properties("${header}" PROPERTIES LANGUAGE "${LANG}")
      cccl_create_clang_tidy_test(hide_from_abi "${header}")
    endforeach()
  endforeach()
endfunction()

cccl_add_abi_tests(CXX ${test_srcs})
enable_language(CUDA)
cccl_add_abi_tests(CUDA ${cu_test_srcs})
