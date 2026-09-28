if (NOT cudax_ENABLE_NCCL)
  return()
endif()

file(GLOB bench_srcs CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

add_bench_dir(bench_targets ${bench_srcs})

foreach (bench_target IN LISTS bench_targets)
  target_link_libraries(${bench_target} PRIVATE NCCL::nccl)
endforeach()
