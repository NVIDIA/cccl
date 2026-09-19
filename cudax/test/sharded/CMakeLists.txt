set(
  sharded_test_sources
  include_only.cu
  concepts/models.cu
  concepts/foreign_models.cu
  containers/sharded_array.cu
  containers/shard_span.cu
  containers/contiguous.cu
  containers/fork_join.cu
  algorithms/elementwise.cu
  algorithms/for_each_shard.cu
  algorithms/generic_tier.cu
  algorithms/reduce_into.cu
  algorithms/reduce_lanes.cu
  algorithms/reduce_scan.cu
  algorithms/zip_transform.cu
  algorithms/count_histogram.cu
  algorithms/compaction.cu
  algorithms/composition_verbs.cu
  algorithms/segmented_reduce.cu
  algorithms/sort.cu
  algorithms/mgmn_adapter.cu
  algorithms/mgmn_std_execution.cu
  algorithms/legacy_parity.cu
  algorithms/engine_parity.cu
  algorithms/bit_determinism.cu
  graph_capture/elementwise_pipeline.cu
  graph_capture/reduce_scan_capture.cu
  graph_capture/must_not_capture.cu
  graph_capture/graph_owned_memory.cu
  graph_capture/contiguous_capture.cu
  graph_capture/mgmn_pipeline_capture.cu
  sparse/sharded_csr.cu
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

## cudax_add_sharded_standalone_header
#
# Add a compile-only executable that includes just `header` (a path relative
# to `cudax/include`), and register it with ctest. The target is named after
# the header's path below `__sharded/`, with `/` replaced by `.`.
#
function(cudax_add_sharded_standalone_header target_name_var header)
  string(REPLACE "cuda/experimental/__sharded/" "" test_label "${header}")
  string(REPLACE ".cuh" "" test_label "${test_label}")
  string(REPLACE "/" "." test_label "${test_label}")

  set(test_target cudax.test.sharded.header_standalone.${test_label})

  cccl_add_executable(${test_target} SOURCES header_standalone.cu ADD_CTEST)
  cudax_places_configure_target(${test_target})
  target_link_libraries(${test_target} PRIVATE cudax.compiler_interface)
  target_compile_definitions(
    ${test_target}
    PRIVATE "CUDAX_SHARDED_HEADER=<${header}>"
  )

  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

foreach (source IN LISTS sharded_test_sources)
  cudax_add_sharded_test(test_target "${source}")
endforeach()

# Vendor tier: cuRAND (algorithm/random/random.cuh is opt-in and not part of
# the umbrella).
cudax_add_sharded_test(curand_test_target "algorithms/random_fill.cu")
target_link_libraries(${curand_test_target} PRIVATE CUDA::curand)

cudax_add_sharded_standalone_header(
  curand_header_target
  "cuda/experimental/__sharded/algorithm/random/random.cuh"
)
target_link_libraries(${curand_header_target} PRIVATE CUDA::curand)

# Math-library tier: cuSPARSE (sparse/ is opt-in; the sharded_csr container
# itself is vendor-free and tested in the main list above), behind the same
# switch as STF's cuBLAS/cuSOLVER tests.
if (cudax_ENABLE_CUDASTF_MATHLIBS)
  foreach (
    source
    IN
    ITEMS sparse/spmv_spmm.cu sparse/rebalance.cu sparse/handle_lifecycle.cu
  )
    cudax_add_sharded_test(cusparse_test_target "${source}")
    target_link_libraries(${cusparse_test_target} PRIVATE CUDA::cusparse)
  endforeach()

  foreach (
    header
    IN
    ITEMS
      cuda/experimental/__sharded/sparse/cusparse.cuh
      cuda/experimental/__sharded/sparse/rebalance.cuh
      cuda/experimental/__sharded/sparse/sparse.cuh
      cuda/experimental/__sharded/sparse/spmm.cuh
      cuda/experimental/__sharded/sparse/spmv.cuh
  )
    cudax_add_sharded_standalone_header(cusparse_header_target "${header}")
    target_link_libraries(${cusparse_header_target} PRIVATE CUDA::cusparse)
  endforeach()
endif()

## Header hygiene: every `__sharded` header must compile on its own, so that
## no header silently depends on a name another header happens to introduce
## first. One compile-only target per header, each including just that one.
## Listed explicitly (like `places_unittested_headers`): `__sharded/reference/`
## — the non-shipping comparison implementations — gets no targets and is
## compiled by the tests that include it (`legacy_parity.cu`,
## `engine_parity.cu`); the vendor tiers (`algorithm/random/`, `sparse/`) are
## registered in the gated vendor blocks above.
set(
  sharded_standalone_headers
  cuda/experimental/__sharded/concepts.cuh
  cuda/experimental/__sharded/cuda_safe_call.cuh
  cuda/experimental/__sharded/concepts/env.cuh
  cuda/experimental/__sharded/concepts/guards.cuh
  cuda/experimental/__sharded/concepts/view.cuh
  cuda/experimental/__sharded/container/csr.cuh
  cuda/experimental/__sharded/container/default_envs.cuh
  cuda/experimental/__sharded/container/shard.cuh
  cuda/experimental/__sharded/container/sharded_array.cuh
  cuda/experimental/__sharded/composition/fork_join.cuh
  cuda/experimental/__sharded/composition/pinned_staging.cuh
  cuda/experimental/__sharded/composition/verbs.cuh
  cuda/experimental/__sharded/engine/generic_map.cuh
  cuda/experimental/__sharded/engine/mgmn.cuh
  cuda/experimental/__sharded/algorithm/adjacent_difference/adjacent_difference.cuh
  cuda/experimental/__sharded/algorithm/copy_if/copy_if.cuh
  cuda/experimental/__sharded/algorithm/count/count.cuh
  cuda/experimental/__sharded/algorithm/fill/fill.cuh
  cuda/experimental/__sharded/algorithm/histogram/histogram.cuh
  cuda/experimental/__sharded/algorithm/reduce/reduce.cuh
  cuda/experimental/__sharded/algorithm/scan/scan.cuh
  cuda/experimental/__sharded/algorithm/segmented_reduce/segmented_reduce.cuh
  cuda/experimental/__sharded/algorithm/sort/shared_va.cuh
  cuda/experimental/__sharded/algorithm/sort/sort.cuh
  cuda/experimental/__sharded/algorithm/transform/transform.cuh
  cuda/experimental/__sharded/algorithm/unique/unique.cuh
)

foreach (header IN LISTS sharded_standalone_headers)
  cudax_add_sharded_standalone_header(test_target "${header}")
endforeach()
