foreach(cn_target IN LISTS cudax_TARGETS)
  cudax_get_target_property(config_prefix ${cn_target} PREFIX)

  # Metatarget for the current configuration's tests:
  set(config_meta_target ${config_prefix}.examples)
  add_custom_target(${config_meta_target})
  add_dependencies(${config_prefix}.all ${config_meta_target})
endforeach()

# FIXME: Enable MSVC
if (NOT "MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}" AND
    NOT "NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # STF tests are handled separately:
  add_subdirectory(stf)
endif()
