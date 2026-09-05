set(
  places_example_sources
  thrust_device_data_place_allocator.cu
  sharded_reduce.cu
)

## cudax_add_places_example
#
# Add a places example executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the example
#   target. Useful for modifying the example/target after creation.
# source: The source file for the example.
#
function(cudax_add_places_example target_name_var source)
  get_filename_component(filename ${source} NAME_WE)

  set(example_target cudax.example.places.${filename})

  cccl_add_executable(${example_target} SOURCES ${source} ADD_CTEST)
  cudax_places_configure_target(${example_target})
  target_link_libraries(
    ${example_target}
    PRIVATE #
      cudax.compiler_interface
      cudax.examples.thrust
  )

  set(${target_name_var} ${example_target} PARENT_SCOPE)
endfunction()

foreach (source IN LISTS places_example_sources)
  cudax_add_places_example(example_target "${source}")
endforeach()

# Vendor tier: the cuRAND examples additionally link the toolkit's cuRAND
# (random.cuh is opt-in and not part of the umbrella).
foreach (source IN ITEMS sharded_pi.cu sharded_rejection_sampling.cu)
  cudax_add_places_example(curand_example_target "${source}")
  target_link_libraries(${curand_example_target} PRIVATE CUDA::curand)
endforeach()
