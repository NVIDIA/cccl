set(jit_template "${CMAKE_CURRENT_BINARY_DIR}/jit_template.h")
set(jit_template_src "${CMAKE_CURRENT_BINARY_DIR}/jit_template.cpp")
set(jit_template_depfile "${CMAKE_CURRENT_BINARY_DIR}/jit_template.d")
set(cpp_entry "${CMAKE_CURRENT_LIST_DIR}/jit_entry.h")

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

set(PP_CMD ${CMAKE_CXX_COMPILER})
set(PP_DEPFILE_KW) # becomes "DEPFILE <file>" on GCC/Clang, empty on MSVC
set(PP_ARGS)

# Define macro with correct spelling per compiler
if(MSVC)
  list(APPEND PP_ARGS /D_CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS)
else()
  list(APPEND PP_ARGS -D_CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS)
endif()

# (Optional) add include dirs if needed
# list(APPEND PP_ARGS /I"${CMAKE_CURRENT_LIST_DIR}")
if(MSVC)
  # Optional but safer: make the path Windows-native to avoid any odd parsing.
  file(TO_NATIVE_PATH "${jit_template}" JIT_TEMPLATE_NATIVE)

  list(APPEND PP_ARGS
    /nologo
    /TP # treat .h as C++
    /P # preprocess only
    /Fi${JIT_TEMPLATE_NATIVE} # <-- no quotes here
    "${cpp_entry}"
  )
else()
  list(APPEND PP_ARGS
    -E "${cpp_entry}"
    -o "${jit_template}"
    -MD -MT "${jit_template}"
    -MF "${jit_template_depfile}"
  )
  set(PP_DEPFILE_KW DEPFILE "${jit_template_depfile}")
endif()

add_custom_command(
  OUTPUT "${jit_template}"
  DEPENDS "${cpp_entry}"
  COMMAND "${PP_CMD}" ${PP_ARGS}
  ${PP_DEPFILE_KW}
  VERBATIM
  COMMENT "Generated ${jit_template}."
)

add_custom_command(
  OUTPUT "${jit_template_src}"
  DEPENDS
  "${jit_template}"
  "${CMAKE_CURRENT_LIST_DIR}/template_pre.h.in"
  "${CMAKE_CURRENT_LIST_DIR}/template_post.h.in"

  COMMAND ${CMAKE_COMMAND} -DOUT:FILEPATH=${jit_template_src}
   -DTEMPLATE_PRE:FILEPATH=${CMAKE_CURRENT_LIST_DIR}/template_pre.h.in
   -DJIT_TEMPLATE:FILEPATH=${jit_template}
   -DTEMPLATE_POST:FILEPATH=${CMAKE_CURRENT_LIST_DIR}/template_post.h.in
   -P ${CMAKE_CURRENT_LIST_DIR}/concat_templates.cmake

  VERBATIM
  COMMENT "Generated ${jit_template_src}."
)

set_source_files_properties("${jit_template_src}" PROPERTIES GENERATED TRUE)
add_library(cccl.c.parallel.jit_template OBJECT "${jit_template_src}")
