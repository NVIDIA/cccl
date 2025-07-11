set(jit_template "${CMAKE_CURRENT_BINARY_DIR}/jit_template.h")
set(jit_template_src "${CMAKE_CURRENT_BINARY_DIR}/jit_template.cpp")
set(jit_template_depfile "${CMAKE_CURRENT_BINARY_DIR}/jit_template.d")
set(cpp_entry "${CMAKE_CURRENT_LIST_DIR}/jit_entry.h")

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

add_custom_command(
    OUTPUT "${jit_template}"
    DEPENDS "${cpp_entry}"
    DEPFILE "${jit_template_depfile}"
    COMMAND "${CMAKE_CXX_COMPILER}" "${cpp_entry}" -E -o "${jit_template}"
        -D_CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
        -MD -MT "${jit_template}" -MF "${jit_template_depfile}"
)

add_custom_command(
    OUTPUT "${jit_template_src}"
    DEPENDS
      "${jit_template}"
      "${CMAKE_CURRENT_LIST_DIR}/template_pre.h.in"
      "${CMAKE_CURRENT_LIST_DIR}/template_post.h.in"
    VERBATIM
    COMMAND "${CMAKE_COMMAND}" -E cat --
        "${CMAKE_CURRENT_LIST_DIR}/template_pre.h.in"
        "${jit_template}"
        "${CMAKE_CURRENT_LIST_DIR}/template_post.h.in"
        > "${jit_template_src}"
)
set_source_files_properties("${jit_template_src}" PROPERTIES GENERATED TRUE)

add_library(cccl.c.parallel.jit_template OBJECT "${jit_template_src}")
