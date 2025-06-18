find_package(cudax) # already found, bring in version info.
find_package(Thrust ${cudax_VERSION} EXACT CONFIG
  NO_DEFAULT_PATH # Only check the explicit path in HINTS:
  HINTS "${CCCL_SOURCE_DIR}/lib/cmake/thrust/"
)
thrust_create_target(cudax.examples.thrust)

function(cudax_add_example target_name_var example_src cudax_target)
  cudax_get_target_property(config_prefix ${cudax_target} PREFIX)
  cudax_get_target_property(config_dialect ${cudax_target} DIALECT)

  get_filename_component(example_name ${example_src} NAME_WE)

  # The actual name of the test's target:
  set(example_target ${config_prefix}.example.${example_name})
  set(${target_name_var} ${example_target} PARENT_SCOPE)

  # Related target names:
  set(config_meta_target ${config_prefix}.examples)
  set(example_meta_target cudax.all.example.${example_name})

  add_executable(${example_target} "${example_src}")
  cccl_configure_target(${example_target} DIALECT ${config_dialect})
  target_link_libraries(${example_target} PRIVATE
    ${cudax_target}
    cudax.examples.thrust
  )
  target_compile_options(${example_target} PRIVATE
    "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr --extended-lambda>
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

file(GLOB example_srcs
  RELATIVE "${cudax_SOURCE_DIR}/examples"
  CONFIGURE_DEPENDS
  *.cu *.cpp
)

find_package(CUDAToolkit REQUIRED) # for CUDAToolkit_VERSION

# Example requires pinned_memory_resource.
if(CUDAToolkit_VERSION VERSION_LESS 12.6)
  list(REMOVE_ITEM example_srcs async_buffer_add.cu cub_reduce.cu)
endif()

foreach(cudax_target IN LISTS cudax_TARGETS)
  cudax_get_target_property(config_prefix ${cudax_target} PREFIX)

  # Metatarget for the current configuration's tests:
  set(config_meta_target ${config_prefix}.examples)
  add_custom_target(${config_meta_target})
  add_dependencies(${config_prefix}.all ${config_meta_target})

  foreach (example_src IN LISTS example_srcs)
    cudax_add_example(example_target "${example_src}" ${cudax_target})
    if (${example_src} MATCHES "stdexec")
      # The stream context needs atomic wait/notify, which is only available on sm70 and above.
      get_target_property(cuda_arch_list ${example_target} CUDA_ARCHITECTURES)
      list(REMOVE_ITEM cuda_arch_list "60" "61")
      set_target_properties(${example_target} PROPERTIES CUDA_ARCHITECTURES "${cuda_arch_list}")
    endif()
  endforeach()
endforeach()

# FIXME: Enable MSVC
if (cudax_ENABLE_CUDASTF AND
    NOT "MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # STF examples are handled separately:
  add_subdirectory(stf)
endif()
