# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

cccl_generate_header_tests(cccl.c.parallel.headers c/parallel/include
  DIALECT 20
  GLOBS "cccl/c/*.h"
)
target_link_libraries(cccl.c.parallel.headers PUBLIC cccl.c.parallel)
