set(
  sharded_test_sources
  include_only.cu
  stream_scope.cu
  concepts/models.cu
  concepts/foreign_models.cu
  containers/sharded_array.cu
  containers/contiguous.cu
  containers/fork_join.cu
  algorithms/elementwise.cu
  algorithms/generic_tier.cu
  algorithms/reduce_into.cu
  algorithms/reduce_scan.cu
  algorithms/zip_transform.cu
  algorithms/count_histogram.cu
  algorithms/compaction.cu
  graph_capture/elementwise_pipeline.cu
  graph_capture/reduce_scan_capture.cu
  graph_capture/must_not_capture.cu
  graph_capture/graph_owned_memory.cu
  graph_capture/contiguous_capture.cu
)

cccl_get_cudatoolkit()

## cudax_add_sharded_test
#
# Add a sharded test executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the test
#   target. Useful for adding target information after creation.
# source: The source file for the test.
#
function(cudax_add_sharded_test target_name_var source)
  get_filename_component(dir ${source} DIRECTORY)
  get_filename_component(filename ${source} NAME_WE)
  if (dir)
    set(filename "${dir}/${filename}")
  endif()
  string(REPLACE "/" "." test_name "${filename}")

  set(test_target cudax.test.sharded.${test_name})

  cccl_add_executable(${test_target} SOURCES ${source} ADD_CTEST)
  cudax_places_configure_target(${test_target})
  target_link_libraries(${test_target} PRIVATE cudax.compiler_interface)

  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

foreach (source IN LISTS sharded_test_sources)
  cudax_add_sharded_test(test_target "${source}")
endforeach()

## Header hygiene: every `__sharded` header must compile on its own, so that
## no header silently depends on a name another header happens to introduce
## first. One compile-only target per header, each including just that one.
file(GLOB sharded_headers
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/../../include/cuda/experimental/__sharded"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../include/cuda/experimental/__sharded/*.cuh")

foreach (header IN LISTS sharded_headers)
  get_filename_component(header_name "${header}" NAME_WE)
  set(test_target cudax.test.sharded.header_standalone.${header_name})

  cccl_add_executable(${test_target} SOURCES header_standalone.cu ADD_CTEST)
  cudax_places_configure_target(${test_target})
  target_link_libraries(${test_target} PRIVATE cudax.compiler_interface)
  target_compile_definitions(${test_target}
    PRIVATE "CUDAX_SHARDED_HEADER=<cuda/experimental/__sharded/${header}>")
endforeach()
