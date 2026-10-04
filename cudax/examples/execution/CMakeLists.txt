# Examples for cudax::execution (senders/receivers).
set(execution_example_sources lane_sharded_pipeline.cu lane_localized_stream.cu)

## cudax_add_execution_example
#
# Add an execution example executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the example
#   target. Useful for modifying the example/target after creation.
# source: The source file for the example.
#
function(cudax_add_execution_example target_name_var source)
  get_filename_component(filename ${source} NAME_WE)

  set(example_target cudax.example.execution.${filename})

  cccl_add_executable(${example_target} SOURCES ${source} ADD_CTEST)
  target_link_libraries(
    ${example_target}
    PRIVATE #
      cudax.compiler_interface
      cudax.examples.thrust
  )
  target_compile_options(
    ${example_target}
    PRIVATE
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
  )

  set(${target_name_var} ${example_target} PARENT_SCOPE)
endfunction()

foreach (source IN LISTS execution_example_sources)
  cudax_add_execution_example(example_target "${source}")
endforeach()
