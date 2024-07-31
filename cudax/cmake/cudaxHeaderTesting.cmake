# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

# Meta target for all configs' header builds:
add_custom_target(cudax.all.headers)

file(GLOB_RECURSE headers
  RELATIVE "${cudax_SOURCE_DIR}/include"
  CONFIGURE_DEPENDS1
  "${cudax_SOURCE_DIR}/include/*.cuh"
  "${cudax_SOURCE_DIR}/include/*.h"
)

set(headertest_srcs)
foreach (header IN LISTS headers)
  set(headertest_src "headers/${header}.cu")
  configure_file("${cudax_SOURCE_DIR}/cmake/header_test.in.cu" "${headertest_src}")
  list(APPEND headertest_srcs "${headertest_src}")
endforeach()

function(cudax_add_header_test label definitions)
  foreach(cn_target IN LISTS cudax_TARGETS)
    cudax_get_target_property(config_prefix ${cn_target} PREFIX)

    set(headertest_target ${config_prefix}.headers.${label})
    add_library(${headertest_target} OBJECT ${headertest_srcs})
    target_link_libraries(${headertest_target} PUBLIC ${cn_target})
    target_compile_definitions(${headertest_target} PRIVATE ${definitions}
                               "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
    cudax_clone_target_properties(${headertest_target} ${cn_target})

    add_dependencies(cudax.all.headers ${headertest_target})
    add_dependencies(${config_prefix}.all ${headertest_target})
  endforeach()
endfunction()

cudax_add_header_test(basic "")
