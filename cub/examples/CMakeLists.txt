# Some of the examples include the `cub/test/test_util.h` header, which
# depends on c2h:
cccl_get_c2h()

## cub_add_example
#
# Add an example executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the example
#   target. Useful for modifying the example/target after creation.
# example_name: The name of the example minus "cub.example." For
#   instance, examples/vector.cu will be "vector", and examples/cuda/copy.cu
#   would be "cuda.copy".
# example_src: The source file that implements the example.
#
function(cub_add_example target_name_var example_name example_src)
  # The actual name of the test's target:
  set(example_target cub.example.${example_name})
  set(${target_name_var} ${example_target} PARENT_SCOPE)

  cccl_add_executable(${example_target} SOURCES "${example_src}" ADD_CTEST)
  cub_configure_cuda_target(${example_target} RDC ${CUB_FORCE_RDC})
  target_link_libraries(
    ${example_target}
    PRIVATE #
      cub.compiler_interface
      cccl.c2h
  )
  target_include_directories(
    ${example_target}
    PRIVATE "${CUB_SOURCE_DIR}/examples"
  )
endfunction()

add_subdirectory(block)
add_subdirectory(device)
