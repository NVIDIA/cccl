file(GLOB bench_srcs CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

add_bench_dir(bench_targets ${bench_srcs})
