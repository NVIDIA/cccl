foreach(thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

  set(framework_target ${config_prefix}.test.framework)

  add_library(${framework_target} INTERFACE)
  target_link_libraries(${framework_target} INTERFACE ${thrust_target})
  #target_include_directories(${framework_target} PRIVATE "${Thrust_SOURCE_DIR}/testing")
  thrust_clone_target_properties(${framework_target} ${thrust_target})
  if ("CUDA" STREQUAL "${config_device}")
    thrust_configure_cuda_target(${framework_target} RDC ${THRUST_FORCE_RDC})
  endif()
endforeach()
