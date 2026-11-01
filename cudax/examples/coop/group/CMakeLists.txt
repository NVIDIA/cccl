function(cudax_add_coop_group_example target_name_var source)
  get_filename_component(filename ${source} NAME_WE)

  set(example_target cudax.example.coop.group.${filename})

  cccl_add_executable(${example_target} SOURCES ${source} ADD_CTEST)
  target_link_libraries(${example_target} PRIVATE cudax.compiler_interface)

  set(${target_name_var} ${example_target} PARENT_SCOPE)
endfunction()

file(
  GLOB example_srcs
  RELATIVE "${cudax_SOURCE_DIR}/examples/coop/group"
  CONFIGURE_DEPENDS
  *.cu
  *.cpp
)

foreach (example_src IN LISTS example_srcs)
  cudax_add_coop_group_example(example_target "${example_src}")
endforeach()
