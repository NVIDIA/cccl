set(
  sharded_test_sources
  include_only.cu
  containers/sharded_array.cu
  containers/contiguous.cu
  containers/fork_join.cu
  algorithms/elementwise.cu
  algorithms/reduce_scan.cu
  algorithms/count_histogram.cu
  algorithms/compaction.cu
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
