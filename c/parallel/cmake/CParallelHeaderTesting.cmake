# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

set(target_name cccl.c.parallel.headers)

cccl_generate_header_tests(${target_name} c/parallel/include
  LANGUAGE C
  GLOBS "cccl/c/*.h"
)
target_link_libraries(${target_name} PUBLIC cccl.c.parallel)
target_include_directories(${target_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
