cccl_generate_install_rules(libcudacxx ${CCCL_TOPLEVEL_PROJECT}
  HEADERS_SUBDIRS "include/cuda" "include/nv"
  HEADERS_INCLUDE "*"
  HEADERS_EXCLUDE "CMakeLists.txt"
  PACKAGE
)
