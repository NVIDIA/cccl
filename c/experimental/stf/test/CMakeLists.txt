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

  cccl_add_executable(
    ${target_name}
    ADD_CTEST
    NO_METATARGETS
    DIALECT 20
    SOURCES "${source}"
  )

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
  )
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
