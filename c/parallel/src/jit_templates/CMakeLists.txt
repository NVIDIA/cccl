set(jit_template "${CMAKE_CURRENT_BINARY_DIR}/jit_template.h")
set(jit_template_src "${CMAKE_CURRENT_BINARY_DIR}/jit_template.cpp")
set(jit_template_depfile "${CMAKE_CURRENT_BINARY_DIR}/jit_template.d")
set(cpp_entry "${CMAKE_CURRENT_LIST_DIR}/jit_entry.h")

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# The pre-processor (PP) command differs between MSVC and others.
set(PP_CMD ${CMAKE_CXX_COMPILER})
set(PP_DEPFILE_KW)
set(PP_ARGS)

if (MSVC)
  list(APPEND PP_ARGS /D_CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS)
else()
  list(APPEND PP_ARGS -D_CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS)
endif()

if (MSVC)
  file(TO_NATIVE_PATH "${jit_template}" JIT_TEMPLATE_NATIVE)

  list(
    APPEND PP_ARGS
    /nologo
    /TP
    /P
    /Fi${JIT_TEMPLATE_NATIVE}
    "${cpp_entry}"
  )
else()
  list(
    APPEND PP_ARGS
    # gersemi: off
    -E "${cpp_entry}"
    -o "${jit_template}"
    -MD -MT "${jit_template}"
    -MF "${jit_template_depfile}"
    # gersemi: on
  )
  set(PP_DEPFILE_KW DEPFILE "${jit_template_depfile}")
endif()

add_custom_command(
  OUTPUT "${jit_template}"
  DEPENDS "${cpp_entry}"
  COMMAND "${PP_CMD}" ${PP_ARGS} ${PP_DEPFILE_KW}
  VERBATIM
  COMMENT "Generated ${jit_template}."
)

add_custom_command(
  OUTPUT "${jit_template_src}"
  DEPENDS
    "${jit_template}"
    "${CMAKE_CURRENT_LIST_DIR}/template_pre.h.in"
    "${CMAKE_CURRENT_LIST_DIR}/template_post.h.in"
  # gersemi: off
  COMMAND
    ${CMAKE_COMMAND}
      -DOUT:FILEPATH=${jit_template_src}
      -DTEMPLATE_PRE:FILEPATH=${CMAKE_CURRENT_LIST_DIR}/template_pre.h.in
      -DJIT_TEMPLATE:FILEPATH=${jit_template}
      -DTEMPLATE_POST:FILEPATH=${CMAKE_CURRENT_LIST_DIR}/template_post.h.in
      -P ${CMAKE_CURRENT_LIST_DIR}/concat_templates.cmake
  # gersemi: on
  VERBATIM
  COMMENT "Generated ${jit_template_src}."
)

set_source_files_properties("${jit_template_src}" PROPERTIES GENERATED TRUE)
add_library(cccl.c.parallel.jit_template OBJECT "${jit_template_src}")
