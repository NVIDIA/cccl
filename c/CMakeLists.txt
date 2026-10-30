if (CCCL_ENABLE_C_PARALLEL AND CCCL_ENABLE_C_PARALLEL_V2)
  message(
    FATAL_ERROR
    "CCCL_ENABLE_C_PARALLEL and CCCL_ENABLE_C_PARALLEL_V2 are mutually exclusive. "
    "v2 is the HostJIT-based successor of v1; pick one."
  )
endif()

if (CCCL_ENABLE_C_PARALLEL)
  add_subdirectory(parallel)
endif()

if (CCCL_ENABLE_C_PARALLEL_V2)
  add_subdirectory(parallel.v2)
endif()

if (CCCL_ENABLE_C_EXPERIMENTAL_STF)
  add_subdirectory(experimental/stf)
endif()
