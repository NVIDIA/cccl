set(stf_example_sources
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
set(stf_example_codegen_sources
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
set(stf_example_mathlib_sources
  linear_algebra/06-pdgemm.cu
  linear_algebra/07-cholesky.cu
  linear_algebra/07-potri.cu
  linear_algebra/cg_csr.cu
  linear_algebra/cg_dense_2D.cu
  linear_algebra/strassen.cu
)

find_package(cudax) # already found, bring in version info.

find_package(CUB ${cudax_VERSION} EXACT CONFIG
  NO_DEFAULT_PATH # Only check the explicit path in HINTS:
  HINTS "${CCCL_SOURCE_DIR}/lib/cmake/cub/"
)

find_package(CUDAToolkit REQUIRED)

## cudax_add_stf_example
#
# Add an stf example executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the example
#   target. Useful for post-processing target information.
# source: The source file for the example.
# cn_target: The reference cudax target with configuration information.
# Additional args are passed to cudax_stf_configure_target.
function(cudax_add_stf_example target_name_var source cn_target)
  cudax_get_target_property(config_dialect ${cn_target} DIALECT)
  cudax_get_target_property(config_prefix ${cn_target} PREFIX)

  get_filename_component(dir ${source} DIRECTORY)
  get_filename_component(filename ${source} NAME_WE)
  if (dir)
    set(filename "${dir}/${filename}")
  endif()
  string(REPLACE "/" "." example_name "stf/${filename}")

  set(example_target ${config_prefix}.example.${example_name})

  add_executable(${example_target} ${source})
  cccl_configure_target(${example_target} DIALECT ${config_dialect})
  cudax_clone_target_properties(${example_target} ${cn_target})
  cudax_stf_configure_target(${example_target} ${ARGN})
  target_link_libraries(${example_target} PRIVATE
    cudax.examples.thrust
    CUB::CUB
  )

  set(stf_meta_target ${config_prefix}.examples.stf)
  add_dependencies(${stf_meta_target} ${example_target})

  add_test(NAME ${example_target} COMMAND "$<TARGET_FILE:${example_target}>")

  set(${target_name_var} ${example_target} PARENT_SCOPE)
endfunction()

# Create examples for each enabled configuration:
foreach(cn_target IN LISTS cudax_TARGETS)
  cudax_get_target_property(config_prefix ${cn_target} PREFIX)

  # Metatarget for the current configuration's examples:
  set(config_meta_target ${config_prefix}.examples)
  set(stf_meta_target ${config_prefix}.examples.stf)
  add_custom_target(${stf_meta_target})
  add_dependencies(${config_meta_target} ${stf_meta_target})

  foreach(source IN LISTS stf_example_sources)
    cudax_add_stf_example(example_target "${source}" ${cn_target})
  endforeach()

  if (cudax_ENABLE_CUDASTF_CODE_GENERATION)
     foreach(source IN LISTS stf_example_codegen_sources)
         cudax_add_stf_example(example_target "${source}" ${cn_target})
     endforeach()
  endif()

  if (cudax_ENABLE_CUDASTF_MATHLIBS)
    foreach(source IN LISTS stf_example_mathlib_sources)
      cudax_add_stf_example(example_target "${source}" ${cn_target} LINK_MATHLIBS)
    endforeach()
  endif()
endforeach()
