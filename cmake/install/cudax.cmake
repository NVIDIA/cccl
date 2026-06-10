cccl_generate_install_rules(
  cudax
  ${CCCL_ENABLE_CUDAX}
  HEADERS_SUBDIRS "include/cuda"
  HEADERS_INCLUDE "*.cuh"
  PACKAGE
)
