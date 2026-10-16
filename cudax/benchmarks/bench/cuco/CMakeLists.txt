file(GLOB bench_srcs CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

add_bench_dir(bench_targets ${bench_srcs})

add_subdirectory(fixed_capacity_map)

add_subdirectory(bloom_filter)

option(
  CCCL_CUDAX_CUCO_BLOOM_FILTER_ENABLE_PROFILING
  "Collect CUPTI DRAM throughput and L2 hit rates in Bloom filter benchmarks"
  OFF
)
target_compile_definitions(
  cudax.bench.cuco.bloom_filter.contains.base
  PRIVATE
    CCCL_CUDAX_CUCO_BLOOM_FILTER_ENABLE_PROFILING=$<BOOL:${CCCL_CUDAX_CUCO_BLOOM_FILTER_ENABLE_PROFILING}>
)
