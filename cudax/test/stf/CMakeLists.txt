set(stf_test_sources
  allocators/buddy_allocator.cu
  cpp/concurrency_test.cu
  cpp/redundant_data.cu
  cpp/redundant_data_different_modes.cu
  cpp/scoped_graph_task.cu
  cpp/user_streams.cu
  dot/basic.cu
  dot/graph_print_to_dot.cu
  dot/sections.cu
  dot/sections_2.cu
  dot/section_movable.cu
  dot/with_events.cu
  error_checks/ctx_mismatch.cu
  error_checks/data_interface_mismatch.cu
  error_checks/double_finalize.cu
  error_checks/erase_frozen.cu
  error_checks/misformed_tasks_dbl_end.cu
  error_checks/misformed_tasks_dbl_start.cu
  error_checks/non_managed_data.cu
  error_checks/uninitialized_data.cu
  error_checks/write_frozen.cu
  examples/cuda_kernels_driver.cu
  examples/05-stencil-no-copy.cu
  examples/05-stencil-places.cu
  examples/05-stencil.cu
  examples/05-stencil2d-places.cu
  fhe/parse_arctyrex.cu
  gnu/include_only.cpp
  graph/concurrency_test.cu
  graph/explicit_graph.cu
  graph/explicit_graph_while.cu
  graph/explicit_graph_while-kernels.cu
  graph/get_cache_stats.cu
  graph/graph_cache_policy.cu
  graph/graph_ctx_low_level.cu
  graph/static_graph_ctx.cu
  hashtable/test.cu
  interface/cuda_kernel_chain-add_deps.cu
  interface/cuda_kernel_chain-add_deps_low_level.cu
  interface/cuda_kernel_empty_args.cu
  interface/data_from_device_async.cu
  interface/move_operator.cu
  local_stf/legacy_to_stf.cu
  local_stf/threads_multiple_graphs.cu
  places/affinity_gc.cu
  places/managed.cu
  places/managed_from_user.cu
  places/non_current_device.cu
  places/place_partition.cu
  places/recursion.cu
  reclaiming/graph.cu
  reclaiming/graph_2.cu
  reclaiming/graph_real_oom.cu
  reclaiming/stream.cu
  reductions/many_inc.cu
  reductions/redux_test.cu
  reductions/redux_test2.cu
  reductions/slice2d_reduction.cu
  reductions/slice_custom_op.cu
  reductions/successive_reductions.cu
  reductions/sum.cu
  reductions/sum_array.cu
  reductions/sum_multiple_places_no_refvalue.cu
  slice/pinning.cu
  stencil/stencil-1D.cu
  stress/empty_tasks.cu
  stress/empty_tasks_alloc.cu
  stress/kernel_chain.cu
  stress/kernel_chain_fused.cu
  stress/many_read.cu
  stress/task_bench.cu
  threads/axpy-threads-2.cu
  # threads/axpy-threads.cu
  utility/timing_with_fences.cu
  utility/source_location_map.cu
)

set(stf_test_codegen_sources
  # algorithm/algorithm_with_read.cu
  # algorithm/graph_algorithms.cu
  # algorithm/in_graph_ctx.cu
  # algorithm/nested.cu
  allocators/adapter.cu
  allocators/cap_tmp_buffers.cu
  cpp/read_const.cu
  cpp/reuse_computation.cu
  cpp/reuse_computation_2.cu
  error_checks/slice_check_bounds.cu
  error_checks/unsatisfiable_spec.cu
  examples/01-axpy-launch-ranges-cg.cu
  examples/01-axpy-places.cu
  # examples/09-nbody-algorithm.cu
  examples/09-nbody-blocked.cu
  examples/09-nbody.cu
  freeze/constant_logical_data.cu
  freeze/freeze.cu
  freeze/freeze_rw.cu
  freeze/task_fence.cu
  freeze/token.cu
  graph/epoch.cu
  # graph/for_each_batched.cu
  # graph/for_each_batched_write.cu
  graph/freeze_for_graph.cu
  # graph/graph_composition.cu
  graph/graph_tmp_data.cu
  graph/many.cu
  graph/multiple_graph_ctx.cu
  green_context/axpy_gc.cu
  green_context/cuda_graph.cu
  green_context/gc_grid.cu
  hash/ctx_hash.cu
  hash/logical_data.cu
  hashtable/fusion.cu
  hashtable/fusion_reduction.cu
  hashtable/parallel_for.cu
  hashtable/parallel_for_shape.cu
  interface/data_from_device.cu
  interface/data_from_device_2.cu
  interface/data_from_device_wb.cu
  interface/graph_use_device_data.cu
  interface/mix_stream_and_graph.cu
  interface/mix_stream_and_graph_2.cu
  interface/scal.cu
  interface/scalar_div.cu
  interface/scalar_interface.cu
  interface/stream_add_callback.cu
  local_stf/interop_cuda.cu
  loop_dispatch/dispatch_on_streams.cu
  loop_dispatch/loop_dispatch.cu
  loop_dispatch/nested_loop_dispatch.cu
  parallel_for/empty_shape_reduce.cu
  parallel_for/fdtd.cu
  parallel_for/parallel_for_all_devs.cu
  parallel_for/parallel_for_box.cu
  parallel_for/parallel_for_repeat.cu
  parallel_for/test2_parallel_for_context.cu
  parallel_for/tiled_loops.cu
  parallel_for/parallel_for_host.cu
  places/cuda_stream_place.cu
  places/managed_from_shape.cu
  reductions/reduce_sum.cu
  reductions/successive_reductions_pfor.cu
  reductions/sum_multiple_places.cu
  reductions/write_back_after_redux.cu
  stress/launch_overhead.cu
  stress/launch_vs_parallelfor.cu
  stress/parallel_for_overhead.cu
  # threads/axpy-threads-pfor.cu   # Currently has a difficult-to-reproduce concurrency problem
  threads/axpy-threads-graph.cu
  threads/axpy-threads-graph-capture.cu
)

# Examples using CUBLAS, CUSOLVER...
set(stf_test_mathlib_sources
  cuda-samples/0_Introduction/vectorAdd/vectorAdd_cudastf.cu
  # Reduce compilation time by not adding this (useless) example
  # cuda-samples/0_Introduction/vectorAdd/vectorAdd
  cuda-samples/3_CUDA_Features/jacobiCudaGraphs/jacobi.cu
  cuda-samples/3_CUDA_Features/jacobiCudaGraphs/jacobi_cudastf.cu
  cuda-samples/4_CUDA_Libraries/conjugateGradientMultiDeviceCG/conjugateGradientMultiDeviceCG_custf.cu
  cuda-samples/5_Domain_Specific/MonteCarloMultiGPU_cudastf/MonteCarloMultiGPU.cu
  examples/07-cholesky-redux.cu
  examples/07-cholesky-unified.cu
  gnu/06-pdgemm.cpp
  gnu/07-cholesky.cpp
)

set(stf_unittested_headers
  cuda/experimental/__stf/allocators/buddy_allocator.cuh
  cuda/experimental/__stf/graph/graph_ctx.cuh
  cuda/experimental/__stf/internal/async_resources_handle.cuh
  cuda/experimental/__stf/internal/context.cuh
  cuda/experimental/__stf/internal/execution_policy.cuh
  cuda/experimental/__stf/internal/interpreted_execution_policy.cuh
  cuda/experimental/__stf/internal/logical_data.cuh
  cuda/experimental/__stf/internal/parallel_for_scope.cuh
  cuda/experimental/__stf/internal/slice.cuh
  cuda/experimental/__stf/internal/thread_hierarchy.cuh
  cuda/experimental/__stf/places/cyclic_shape.cuh
  cuda/experimental/__stf/places/inner_shape.cuh
  cuda/experimental/__stf/places/places.cuh
  cuda/experimental/__stf/places/tiled_partition.cuh
  cuda/experimental/__stf/stream/stream_ctx.cuh
  cuda/experimental/__stf/utility/cartesian_iterator.cuh
  cuda/experimental/__stf/utility/cuda_safe_call.cuh
  cuda/experimental/__stf/utility/dimensions.cuh
  cuda/experimental/__stf/utility/handle.cuh
  cuda/experimental/__stf/utility/hash.cuh
  cuda/experimental/__stf/utility/memory.cuh
  cuda/experimental/__stf/utility/scope_guard.cuh
  cuda/experimental/__stf/utility/stopwatch.cuh
  cuda/experimental/__stf/utility/unittest.cuh
  cuda/experimental/__stf/utility/unstable_unique.cuh
)

find_package(CUDAToolkit REQUIRED)

## cudax_add_stf_test
#
# Add an stf test executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the test
#   target. Useful for post-processing target information.
# source: The source file for the test.
# cn_target: The reference cudax target with configuration information.
# Additional args are passed to cudax_stf_configure_target.
function(cudax_add_stf_test target_name_var source cn_target)
  cudax_get_target_property(config_dialect ${cn_target} DIALECT)
  cudax_get_target_property(config_prefix ${cn_target} PREFIX)

  get_filename_component(dir ${source} DIRECTORY)
  get_filename_component(filename ${source} NAME_WE)
  if (dir)
    set(filename "${dir}/${filename}")
  endif()
  string(REPLACE "/" "." test_name "${filename}")

  set(test_target ${config_prefix}.test.stf.${test_name})

  add_executable(${test_target} ${source})
  cccl_configure_target(${test_target} DIALECT ${config_dialect})
  cudax_clone_target_properties(${test_target} ${cn_target})
  cudax_stf_configure_target(${test_target} ${ARGN})

  set(stf_meta_target ${config_prefix}.tests.stf)
  add_dependencies(${stf_meta_target} ${test_target})

  add_test(NAME ${test_target} COMMAND "$<TARGET_FILE:${test_target}>")

  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

## cudax_add_stf_unittest_header
#
# Add an stf unittested header executable and register it with ctest.
#
# Unittested headers contain a set of tests that are enabled by including
# `unittest.cuh` and defining `UNITTESTED_FILE`.
#
# target_name_var: Variable name to overwrite with the name of the test
#   target. Useful for post-processing target information.
# source: The source file for the test.
# cn_target: The reference cudax target with configuration information.
# Additional args are passed to cudax_stf_configure_target.
function(cudax_add_stf_unittest_header target_name_var source cn_target)
  cudax_get_target_property(config_dialect ${cn_target} DIALECT)
  cudax_get_target_property(config_prefix ${cn_target} PREFIX)

  get_filename_component(relative_path ${source} DIRECTORY)
  get_filename_component(filename ${source} NAME_WE)

  string(REPLACE "cuda/experimental/" "" test_label "${relative_path}/${filename}")
  string(REPLACE "/" "." test_label "${test_label}")
  set(test_target "${config_prefix}.test.stf.unittest_headers.${test_label}")

  # Pass the full path to configure_file (this is configured from cudax/tests/stf/
  get_filename_component(source_full_path ../../../cudax/include/${source} ABSOLUTE)
  set(source ${source_full_path})

  set(ut_template "${cudax_SOURCE_DIR}/cmake/stf_header_unittest.in.cu")
  set(ut_source "${cudax_BINARY_DIR}/unittest_headers/${test_target}.cu")
  configure_file(${ut_template} ${ut_source} @ONLY)

  add_executable(${test_target} ${ut_source})
  cccl_configure_target(${test_target} DIALECT ${config_dialect})
  cudax_clone_target_properties(${test_target} ${cn_target})
  cudax_stf_configure_target(${test_target} ${ARGN})

  set(stf_unittest_headers_meta_target ${config_prefix}.tests.stf.unittest_headers)
  add_dependencies(${stf_unittest_headers_meta_target} ${test_target})

  add_test(NAME ${test_target} COMMAND ${test_target})

  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

# Create tests for each enabled configuration:
foreach(cn_target IN LISTS cudax_TARGETS)
  cudax_get_target_property(config_prefix ${cn_target} PREFIX)

  # Metatargets for the current configuration's tests:
  set(config_meta_target ${config_prefix}.tests)
  set(stf_test_meta_target ${config_prefix}.tests.stf)
  add_custom_target(${stf_test_meta_target})
  add_dependencies(${config_meta_target} ${stf_test_meta_target})
  set(stf_unittest_headers_meta_target ${config_prefix}.tests.stf.unittest_headers)
  add_custom_target(${stf_unittest_headers_meta_target})
  add_dependencies(${stf_test_meta_target} ${stf_unittest_headers_meta_target})

  # Basic tests:
  foreach(source IN LISTS stf_test_sources)
    cudax_add_stf_test(test_target "${source}" ${cn_target})
  endforeach()

  if (cudax_ENABLE_CUDASTF_CODE_GENERATION)
    foreach(source IN LISTS stf_test_codegen_sources)
      cudax_add_stf_test(test_target "${source}" ${cn_target})
    endforeach()
  endif()

  # Tests with mathlib deps:
  if (cudax_ENABLE_CUDASTF_MATHLIBS)
    foreach(source IN LISTS stf_test_mathlib_sources)
      cudax_add_stf_test(test_target "${source}" ${cn_target} LINK_MATHLIBS)
    endforeach()
  endif()

  # Unittested headers
  foreach(source IN LISTS stf_unittested_headers)
    cudax_add_stf_unittest_header(test_target "${source}" ${cn_target})
  endforeach()
endforeach()

add_subdirectory(static_error_checks)
