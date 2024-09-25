cccl_generate_install_rules(cudax ${CCCL_TOPLEVEL_PROJECT}
  HEADERS_SUBDIRS "include/cuda"
  HEADERS_INCLUDE "*.cuh"
  PACKAGE
)
