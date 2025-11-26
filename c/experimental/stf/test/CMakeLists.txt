cccl_get_c2h()

function(cccl_c_experimental_stf_add_test target_name_var source)
  string(
    REGEX REPLACE
    "test_([^.]*)"
    "cccl.c.experimental.stf.test.\\1"
    target_name
    "${source}"
  )
  set(target_name_var ${target_name} PARENT_SCOPE)

  add_executable(${target_name} "${source}")
  cccl_configure_target(${target_name} DIALECT 20)

  set_target_properties(${target_name} PROPERTIES CUDA_RUNTIME_LIBRARY STATIC)
  target_link_libraries(
    ${target_name}
    PRIVATE
      cccl.compiler_interface
      cccl.c.experimental.stf
      CUDA::cudart_static
      CUDA::nvrtc
      cccl.c2h.main
      CUDA::cuda_driver
      CCCL::cudax
  )

  add_test(NAME ${target_name} COMMAND ${target_name})
endfunction()

file(
  GLOB test_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  *.cu
  *.cpp
)

foreach (test_src IN LISTS test_srcs)
  cccl_c_experimental_stf_add_test(test_target "${test_src}")
endforeach()
