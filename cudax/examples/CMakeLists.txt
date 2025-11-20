thrust_create_target(cudax.examples.thrust)

function(cudax_add_example target_name_var example_src)
  get_filename_component(example_name ${example_src} NAME_WE)

  # The actual name of the test's target:
  set(example_target cudax.example.${example_name})
  set(${target_name_var} ${example_target} PARENT_SCOPE)

  cccl_add_executable(${example_target} SOURCES "${example_src}" ADD_CTEST)
  target_link_libraries(
    ${example_target}
    PRIVATE #
      cudax.compiler_interface
      cudax.examples.thrust
  )
  target_compile_definitions(
    ${example_target}
    PRIVATE LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
  )
  target_compile_options(
    ${example_target}
    PRIVATE
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
  )
  target_include_directories(
    ${example_target}
    PRIVATE "${CUB_SOURCE_DIR}/examples"
  )
endfunction()

file(
  GLOB example_srcs
  RELATIVE "${cudax_SOURCE_DIR}/examples"
  CONFIGURE_DEPENDS
  *.cu
  *.cpp
)

find_package(CUDAToolkit REQUIRED) # for CUDAToolkit_VERSION

# Example requires pinned_memory_resource.
if (CUDAToolkit_VERSION VERSION_LESS 12.6)
  list(REMOVE_ITEM example_srcs async_buffer_add.cu cub_reduce.cu)
endif()

foreach (example_src IN LISTS example_srcs)
  cudax_add_example(example_target "${example_src}")
endforeach()

# FIXME: Enable MSVC
if (cudax_ENABLE_CUDASTF AND NOT "MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # STF examples are handled separately:
  add_subdirectory(stf)
endif()
