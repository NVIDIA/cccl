if (CCCL_ENABLE_C_PARALLEL)
  add_subdirectory(parallel)
endif()

if (CCCL_ENABLE_C_EXPERIMENTAL_STF)
  add_subdirectory(experimental/stf)
endif()
