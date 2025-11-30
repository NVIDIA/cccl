set(
  stf_example_sources
  01-axpy.cu
  01-axpy-cuda_kernel.cu
  01-axpy-cuda_kernel_chain.cu
  02-axpy-host_launch.cu
  03-temporary-data.cu
  04-fibonacci.cu
  04-fibonacci-run_once.cu
  08-cub-reduce.cu
  axpy-annotated.cu
  void_data_interface.cu
  explicit_data_places.cu
  thrust_zip_iterator.cu
  1f1b.cu
)

# Examples which rely on code generation (parallel_for or launch)
set(
  stf_example_codegen_sources
  01-axpy-launch.cu
  01-axpy-parallel_for.cu
  binary_fhe.cu
  09-dot-reduce.cu
  cfd.cu
  custom_data_interface.cu
  fdtd_mgpu.cu
  frozen_data_init.cu
  graph_algorithms/degree_centrality.cu
  graph_algorithms/jaccard.cu
  graph_algorithms/pagerank.cu
  graph_algorithms/tricount.cu
  heat.cu
  heat_mgpu.cu
  jacobi.cu
  jacobi_pfor.cu
  launch_histogram.cu
  launch_scan.cu
  launch_sum.cu
  launch_sum_cub.cu
  logical_gates_composition.cu
  mandelbrot.cu
  parallel_for_2D.cu
  pi.cu
  scan.cu
  standalone-launches.cu
  word_count.cu
  word_count_reduce.cu
)

# Examples using CUBLAS, CUSOLVER...
set(
  stf_example_mathlib_sources
  linear_algebra/06-pdgemm.cu
  linear_algebra/07-cholesky.cu
  linear_algebra/07-potri.cu
  linear_algebra/cg_csr.cu
  linear_algebra/cg_dense_2D.cu
  linear_algebra/strassen.cu
)

cccl_get_cudatoolkit()

## cudax_add_stf_example
#
# Add an stf example executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the example
#   target. Useful for modifying the example/target after creation.
# source: The source file for the example.
#
# Additional args are passed to cudax_stf_configure_target.
function(cudax_add_stf_example target_name_var source)
  get_filename_component(dir ${source} DIRECTORY)
  get_filename_component(filename ${source} NAME_WE)
  if (dir)
    set(filename "${dir}/${filename}")
  endif()
  string(REPLACE "/" "." example_name "stf/${filename}")

  set(example_target cudax.example.${example_name})

  cccl_add_executable(${example_target} SOURCES ${source} ADD_CTEST)
  cudax_stf_configure_target(${example_target} ${ARGN})
  target_link_libraries(
    ${example_target}
    PRIVATE #
      cudax.compiler_interface
      cudax.examples.thrust
  )

  set(${target_name_var} ${example_target} PARENT_SCOPE)
endfunction()

foreach (source IN LISTS stf_example_sources)
  cudax_add_stf_example(example_target "${source}")
endforeach()

if (cudax_ENABLE_CUDASTF_CODE_GENERATION)
  foreach (source IN LISTS stf_example_codegen_sources)
    cudax_add_stf_example(example_target "${source}")
  endforeach()
endif()

if (cudax_ENABLE_CUDASTF_MATHLIBS)
  foreach (source IN LISTS stf_example_mathlib_sources)
    cudax_add_stf_example(example_target "${source}" LINK_MATHLIBS)
  endforeach()
endif()
