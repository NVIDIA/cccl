find_package(cudax REQUIRED) # already found, bring in version info.
find_package(CUDAToolkit REQUIRED)

set(legculib_target cudax.example.cccl-rt.legacy_interact.legculib)
add_library(${legculib_target} SHARED legculib.cpp)
cccl_configure_target(${legculib_target} DIALECT 17)
target_link_libraries(${legculib_target}
  PUBLIC
    CUDA::cuda_driver
  PRIVATE
    CUDA::nvrtc
    cudax.compiler_interface_cpp17)

function(cudax_add_cccl_rt_legacy_interact_example target_name_var cudax_target)
  cudax_get_target_property(config_prefix ${cudax_target} PREFIX)
  cudax_get_target_property(config_dialect ${cudax_target} DIALECT)

  set(example_name legacy_interact)

  # The actual name of the test's target:
  set(example_target ${config_prefix}.example.cccl-rt.${example_name})
  set(${target_name_var} ${example_target} PARENT_SCOPE)

  # Related target names:
  set(config_meta_target ${config_prefix}.examples)
  set(example_meta_target cudax.all.example.cccl-rt.${example_name})

  add_executable(${example_target} "main.cpp")
  cccl_configure_target(${example_target} DIALECT ${config_dialect})
  target_link_libraries(${example_target} PRIVATE
    ${cudax_target}
    libcudacxx::libcudacxx
    ${legculib_target}
    CUDA::cudart
  )
  target_compile_options(${example_target} PRIVATE
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
  )
  target_compile_definitions(${example_target} PRIVATE
    "LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"
  )

  cudax_clone_target_properties(${example_target} ${cudax_target})
  target_include_directories(${example_target} PRIVATE "${CUB_SOURCE_DIR}/examples")

  # Add to the active configuration's meta target
  add_dependencies(${config_meta_target} ${example_target})

  # Meta target that builds examples with this name for all configurations:
  if (NOT TARGET ${example_meta_target})
    add_custom_target(${example_meta_target})
  endif()
  add_dependencies(${example_meta_target} ${example_target})

  add_test(NAME ${example_target}
    COMMAND "$<TARGET_FILE:${example_target}>"
  )
endfunction()

foreach(cudax_target IN LISTS cudax_TARGETS)
  cudax_get_target_property(config_prefix ${cudax_target} PREFIX)

  # Metatarget for the current configuration's tests:
  set(config_meta_target ${config_prefix}.examples.cccl-rt)
  add_custom_target(${config_meta_target})
  add_dependencies(${config_prefix}.all ${config_meta_target})

  cudax_add_cccl_rt_legacy_interact_example(example_target ${cudax_target})
endforeach()
