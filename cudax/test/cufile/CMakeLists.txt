cccl_get_c2h()

find_package(cudax) # already found, bring in version info.
find_package(CUDAToolkit REQUIRED)

## cudax_add_catch2_cufile_test
#
# Add a catch2 cufile test executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the test
#   target. Useful for post-processing target information.
# test_name: A unique name for the executable that will be appended to
#  "<config_prefix>.test.".
# cudax_target: The reference cudax target with configuration information.
#
# Additional arguments will be processed as test sources.
#
function(cudax_add_catch2_cufile_test target_name_var test_name cudax_target) # ARGN=test sources
  cudax_get_target_property(config_prefix ${cudax_target} PREFIX)
  cudax_get_target_property(config_dialect ${cudax_target} DIALECT)

  set(test_target ${config_prefix}.test.${test_name})
  set(test_sources ${ARGN})

  add_executable(${test_target} ${test_sources})
  cccl_configure_target(${test_target} DIALECT ${config_dialect})
  cudax_clone_target_properties(${test_target} ${cudax_target})
  target_include_directories(${test_target} PRIVATE "../common")
  target_link_libraries(
    ${test_target}
    PRIVATE #
      ${cudax_target}
      cccl.c2h.main
      CUDA::cudart
      CUDA::cuFile
  )

  set(config_meta_target ${config_prefix}.tests.cufile)
  add_dependencies(${config_meta_target} ${test_target})

  if (cudax_ENABLE_CUFILE_TESTS_RUN)
    add_test(NAME ${test_target} COMMAND "$<TARGET_FILE:${test_target}>")
  endif()

  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

# Create tests for each enabled configuration:
foreach (cudax_target IN LISTS cudax_TARGETS)
  cudax_get_target_property(config_prefix ${cudax_target} PREFIX)

  # Metatarget for the current configuration's tests:
  set(config_meta_target ${config_prefix}.tests.cufile)
  add_custom_target(${config_meta_target})
  add_dependencies(${config_prefix}.tests ${config_meta_target})

  cudax_add_catch2_cufile_test(test_target cufile_driver_attributes ${cudax_target}
    driver_attributes.cu
  )

  cudax_add_catch2_cufile_test(test_target cufile_driver_register ${cudax_target}
    driver_register.cu
  )

  cudax_add_catch2_cufile_test(test_target cufile_cufile ${cudax_target}
    cufile.cu
  )

  cudax_add_catch2_cufile_test(test_target cufile_cufile_ref ${cudax_target}
    cufile_ref.cu
  )

  cudax_add_catch2_cufile_test(test_target cufile_open_mode ${cudax_target}
    open_mode.cu
  )
endforeach()
