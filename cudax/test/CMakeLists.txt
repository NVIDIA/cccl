cccl_get_c2h()

find_package(cudax) # already found, bring in version info.

## cudax_add_test
#
# Add a catch2 test executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the test
#   target. Useful for post-processing target information.
# test_name: A unique name for the executable that will be appended to
#  "<config_prefix>.test.".
# cn_target: The reference cudax target with configuration information.
#
# Additional arguments will be processed as test sources.
#
function(cudax_add_catch2_test target_name_var test_name cn_target) # ARGN=test sources
  cudax_get_target_property(config_prefix ${cn_target} PREFIX)

  set(test_target ${config_prefix}.test.${test_name})
  set(test_sources ${ARGN})

  add_executable(${test_target} ${test_sources})
  target_include_directories(${test_target} PRIVATE "common")
  target_link_libraries(${test_target} PRIVATE
    ${cn_target}
    cccl.c2h.main
  )
  target_compile_options(${test_target} PRIVATE
    "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
  )

  cudax_clone_target_properties(${test_target} ${cn_target})
  set_target_properties(${test_target} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  )

  set(config_meta_target ${config_prefix}.tests)
  add_dependencies(${config_meta_target} ${test_target})

  add_test(NAME ${test_target} COMMAND "$<TARGET_FILE:${test_target}>")

  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

# Create tests for each enabled configuration:
foreach(cn_target IN LISTS cudax_TARGETS)
  cudax_get_target_property(config_prefix ${cn_target} PREFIX)

  # Metatarget for the current configuration's tests:
  set(config_meta_target ${config_prefix}.tests)
  add_custom_target(${config_meta_target})
  add_dependencies(${config_prefix}.all ${config_meta_target})

  # Add tests:
  cudax_add_catch2_test(test_target hierarchy ${cn_target}
    hierarchy/hierarchy_smoke.cu
    hierarchy/hierarchy_custom_types.cu
  )

  cudax_add_catch2_test(test_target launch ${cn_target}
    launch/launch_smoke.cu
  )

  cudax_add_catch2_test(test_target launch_configuration ${cn_target}
    launch/configuration.cu
  )

  cudax_add_catch2_test(test_target execution ${cn_target}
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
    #execution/test_on.cu
    execution/test_sequence.cu
    execution/test_starts_on.cu
    #execution/test_stream_context.cu
    execution/test_then.cu
    execution/test_visit.cu
    execution/test_when_all.cu
    execution/test_write_attrs.cu
    execution/test_write_env.cu
  )
  target_compile_options(${test_target} PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>)
  target_compile_options(${test_target} PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>)

  # The stream context needs atomic wait/notify, which is only available on sm70 and
  # above.
  get_target_property(cuda_arch_list ${test_target} CUDA_ARCHITECTURES)
  list(REMOVE_ITEM cuda_arch_list "60" "61")
  set_target_properties(${test_target} PROPERTIES CUDA_ARCHITECTURES "${cuda_arch_list}")

  cudax_add_catch2_test(test_target graph ${cn_target}
    graph/graph_smoke.cu
  )

  cudax_add_catch2_test(test_target stream ${cn_target}
    stream/stream_smoke.cu
  )

  cudax_add_catch2_test(test_target misc ${cn_target}
    utility/ensure_current_device.cu
  )

  cudax_add_catch2_test(test_target containers ${cn_target}
    containers/uninitialized_buffer.cu
    containers/uninitialized_async_buffer.cu
    containers/async_buffer/access.cu
    containers/async_buffer/capacity.cu
    containers/async_buffer/constructor.cu
    containers/async_buffer/conversion.cu
    containers/async_buffer/copy.cu
    containers/async_buffer/iterators.cu
    containers/async_buffer/properties.cu
    containers/async_buffer/swap.cu
    containers/async_buffer/transform.cu
  )

  cudax_add_catch2_test(test_target cuco ${cn_target}
    cuco/utility/test_hashers.cu
  )

  cudax_add_catch2_test(test_target memory_resource ${cn_target}
    memory_resource/any_async_resource.cu
    memory_resource/any_resource.cu
    memory_resource/memory_pools.cu
    memory_resource/device_memory_resource.cu
    memory_resource/managed_memory_resource.cu
    memory_resource/pinned_memory_resource.cu
    memory_resource/shared_resource.cu
  )

  cudax_add_catch2_test(test_target green_context ${cn_target}
    green_context/green_ctx_smoke.cu
  )

  cudax_add_catch2_test(test_target kernel ${cn_target}
    kernel/kernel_ref.cu
  )

  cudax_add_catch2_test(test_target library_ref ${cn_target}
    library/library_ref.cu
  )

  cudax_add_catch2_test(test_target library ${cn_target}
    library/library.cu
  )

  cudax_add_catch2_test(test_target algorithm ${cn_target}
    algorithm/fill.cu
    algorithm/copy.cu
  )

endforeach()

# FIXME: Enable MSVC
if (cudax_ENABLE_CUDASTF AND
    NOT "MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # STF tests are handled separately:
  add_subdirectory(stf)
endif()
