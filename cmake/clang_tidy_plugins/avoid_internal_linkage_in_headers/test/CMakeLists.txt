list(APPEND CMAKE_MESSAGE_CONTEXT "test")

file(GLOB test_srcs CONFIGURE_DEPENDS "*.hpp")

foreach (src IN LISTS test_srcs)
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
    set_source_files_properties("${header}" PROPERTIES LANGUAGE CXX)
    cccl_create_clang_tidy_test(
      cccl.tidy.avoid_internal_linkage_in_headers
      "${header}"
    )
  endforeach()
endforeach()
