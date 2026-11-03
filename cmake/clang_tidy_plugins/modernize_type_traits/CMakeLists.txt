list(APPEND CMAKE_MESSAGE_CONTEXT "modernize_type_traits")

function(_cccl_create_catalog_generator)
  # Create a mega file which just includes all of the umbrella headers we care about
  # searching so we can run the catalog generator over it. It should flag any missing
  # traits. This is better (and less error-prone) than checking each header one by one.
  #
  # We do not ever intend to compile this file, we just need:
  #
  # 1. Something we can put in compile_commands.json so clang tooling knows how to compile
  #    libcu++.
  # 2. Something that automatically picks up any new header additions in cuda/ or
  #    cuda/std/.
  block(PROPAGATE dummy_file)
    file(
      GLOB cuda_headers
      CONFIGURE_DEPENDS
      LIST_DIRECTORIES FALSE
      RELATIVE "${CCCL_SOURCE_DIR}/libcudacxx/include"
      "${CCCL_SOURCE_DIR}/libcudacxx/include/cuda/*"
      "${CCCL_SOURCE_DIR}/libcudacxx/include/cuda/std/*"
    )

    list(SORT cuda_headers)
    list(TRANSFORM cuda_headers REPLACE "^(.+)$" "#include <\\1>\n")
    list(JOIN cuda_headers "" cuda_headers)

    set(dummy_file "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp")
    file(GENERATE OUTPUT "${dummy_file}" CONTENT "${cuda_headers}")
    set(dummy_target cccl.tidy.dummy_target.no_compile)
    add_library("${dummy_target}" OBJECT EXCLUDE_FROM_ALL "${dummy_file}")
    target_compile_definitions("${dummy_target}" PRIVATE CCCL_ENABLE_FPTOOL)
    target_link_libraries(
      "${dummy_target}"
      PRIVATE libcudacxx::libcudacxx $<COMPILE_ONLY:CUDA::cuda_driver>
    )
  endblock()

  block(PROPAGATE generate_target)
    set(generate_target cccl.test.tidy.generate_trait_catalog)
    add_executable("${generate_target}" generate_trait_catalog.cpp)
    target_include_directories(
      "${generate_target}"
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../
        ${CLANG_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
    )
    target_compile_definitions(
      "${generate_target}"
      PRIVATE ${LLVM_DEFINITIONS_LIST}
    )
    target_link_libraries(
      "${generate_target}"
      PRIVATE ${CCCL_LLVM_LIBS} clangTooling clangASTMatchers
    )
    cccl_ensure_metatargets("${generate_target}")
  endblock()

  add_test(
    NAME "${generate_target}"
    COMMAND
      "${generate_target}" -p "${CMAKE_BINARY_DIR}"
      "--namespaces=cuda,cuda::std" "${dummy_file}"
  )
  _cccl_add_build_fixture("${generate_target}")
  _cccl_test_requires_target("${generate_target}" "${generate_target}")

  add_custom_target(
    cccl.tidy.modernize_type_traits.regenerate_catalog
    COMMAND
      "${generate_target}" -p "${CMAKE_BINARY_DIR}"
      "--namespaces=cuda,cuda::std" "${dummy_file}" --mode=output
      --output="${CMAKE_CURRENT_LIST_DIR}/trait_catalog.h"
    COMMENT "regenerating cccl-modenize-type-traits trait catalog"
  )
endfunction()

cccl_create_clang_tidy_plugin(modernize_type_traits.cpp)

cccl_get_cudatoolkit()

_cccl_create_catalog_generator()

add_subdirectory(test)
