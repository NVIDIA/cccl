cccl_get_catch2()

find_package(Thrust ${cudax_VERSION} EXACT CONFIG
  NO_DEFAULT_PATH # Only check the explicit path in HINTS:
  HINTS "${CCCL_SOURCE_DIR}/lib/cmake/thrust/"
)
thrust_create_target(cudax.test.thrust)

add_library(catch2_main STATIC catch2_helpers/catch2_main.cpp)
target_link_libraries(catch2_main PUBLIC Catch2::Catch2)

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
    cudax.test.thrust
    Catch2::Catch2
    catch2_main
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
    launch/configuration.cu
  )

  cudax_add_catch2_test(test_target device ${cn_target}
    device/device_smoke.cu
    device/arch_traits.cu
  )
  target_compile_options(${test_target} PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>)
  target_compile_options(${test_target} PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>)

  cudax_add_catch2_test(test_target event ${cn_target}
    event/event_smoke.cu
  )
  target_compile_options(${test_target} PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>)

  cudax_add_catch2_test(test_target stream ${cn_target}
    stream/get_stream.cu
    stream/stream_smoke.cu
  )

  cudax_add_catch2_test(test_target misc ${cn_target}
    utility/driver_api.cu
    utility/ensure_current_device.cu
  )

  cudax_add_catch2_test(test_target containers ${cn_target}
    containers/uninitialized_buffer.cu
    containers/uninitialized_async_buffer.cu
  )

  cudax_add_catch2_test(test_target memory_resource ${cn_target}
    memory_resource/any_async_resource.cu
    memory_resource/any_resource.cu
    memory_resource/async_memory_pool.cu
    memory_resource/async_memory_resource.cu
    memory_resource/shared_resource.cu
  )

  cudax_add_catch2_test(test_target async ${cn_target}
    async/test_conditional.cu
    async/test_continue_on.cu
    async/test_just.cu
    async/test_sequence.cu
    async/test_when_all.cu
  )
  target_compile_options(${test_target} PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>)
  target_compile_options(${test_target} PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>)

  cudax_add_catch2_test(test_target green_context ${cn_target}
    green_context/green_ctx_smoke.cu
  )

endforeach()
