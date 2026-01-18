cccl_get_c2h()
cccl_get_cudatoolkit()

## cudax_add_test
#
# Add a catch2 test executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the test
#   target. Useful for modifying the test/target after creation.
# test_name: A unique name for the executable that will be appended to "cudax.test.".
#
# Additional arguments will be processed as test sources.
#
function(cudax_add_catch2_test target_name_var test_name) # ARGN=test sources
  set(test_target cudax.test.${test_name})
  set(test_sources ${ARGN})

  cccl_add_executable(${test_target} SOURCES ${ARGN} ADD_CTEST)
  target_include_directories(${test_target} PRIVATE "common")
  target_link_libraries(
    ${test_target}
    PRIVATE #
      cudax.compiler_interface
      cccl.c2h.main
      CUDA::cudart_static
  )
  target_compile_options(
    ${test_target}
    PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
  )

  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

cudax_add_catch2_test(test_target launch
    launch/launch_smoke.cu
)

cudax_add_catch2_test(test_target execution ${cudax_target}
    execution/env.cu
    execution/policies/policies.cu
    execution/policies/get_execution_policy.cu
    execution/test_bulk.cu
    execution/test_concepts.cu
    execution/test_completion_signatures.cu
    execution/test_conditional.cu
    execution/test_continues_on.cu
    execution/test_just.cu
    execution/test_let_value.cu
    execution/test_on.cu
    execution/test_sequence.cu
    execution/test_starts_on.cu
    execution/test_stream_context.cu
    execution/test_task_scheduler.cu
    execution/test_then.cu
    execution/test_trampoline_scheduler.cu
    execution/test_visit.cu
    execution/test_when_all.cu
    execution/test_write_attrs.cu
    execution/test_write_env.cu
)

target_compile_options(
  ${test_target}
  PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-allow-unsupported-compiler>
)

cudax_add_catch2_test(test_target graph
    graph/graph_smoke.cu
)

cudax_add_catch2_test(test_target stream
    stream/stream_smoke.cu
)

cudax_add_catch2_test(test_target misc
    utility/ensure_current_device.cu
)

cudax_add_catch2_test(test_target containers
    containers/uninitialized_buffer.cu
)

cudax_add_catch2_test(test_target cuco
    cuco/utility/test_hashers.cu
)

cudax_add_catch2_test(test_target green_context
    green_context/green_ctx_smoke.cu
)

cudax_add_catch2_test(test_target kernel
    kernel/kernel_ref.cu
)

cudax_add_catch2_test(test_target library_ref
    library/library_ref.cu
)

cudax_add_catch2_test(test_target library
    library/library.cu
)

cudax_add_catch2_test(test_target algorithm
    algorithm/fill.cu
    algorithm/copy.cu
)

if (cudax_ENABLE_CUFILE)
  cudax_add_catch2_test(test_target cufile.driver_attributes
      cufile/driver_attributes.cu
  )
  target_link_libraries(${test_target} PRIVATE CUDA::cuFile)

  cudax_add_catch2_test(test_target cufile.driver_register
      cufile/driver_register.cu
  )
  target_link_libraries(${test_target} PRIVATE CUDA::cuFile)

  cudax_add_catch2_test(test_target cufile.cufile
      cufile/cufile.cu
  )
  target_link_libraries(${test_target} PRIVATE CUDA::cuFile)

  cudax_add_catch2_test(test_target cufile.cufile_ref
      cufile/cufile_ref.cu
  )
  target_link_libraries(${test_target} PRIVATE CUDA::cuFile)

  cudax_add_catch2_test(test_target cufile.open_mode
      cufile/open_mode.cu
  )
  target_link_libraries(${test_target} PRIVATE CUDA::cuFile)
endif()

# FIXME: Enable MSVC
if (cudax_ENABLE_CUDASTF AND NOT "MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # STF tests are handled separately:
  add_subdirectory(stf)
endif()
