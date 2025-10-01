cccl_get_c2h()

function(cccl_c_parallel_add_test target_name_var source)
  string(REGEX REPLACE "test_([^.]*)" "cccl.c.parallel.test.\\1" target_name "${source}")
  set(target_name_var ${target_name} PARENT_SCOPE)

  add_executable(${target_name} "${source}")
  cccl_configure_target(${target_name} DIALECT 20)

  set_target_properties(${target_name} PROPERTIES CUDA_RUNTIME_LIBRARY STATIC)
  target_link_libraries(${target_name} PRIVATE
    cccl.c.parallel
    CUDA::cudart_static
    CUDA::nvrtc
    cccl.c2h.main
    cccl.compiler_interface_cpp20
  )

  # Get the first CUDA include directory only
  list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_FIRST_INCLUDE_DIR)
  target_compile_definitions(${target_name} PRIVATE
    TEST_CUB_PATH="-I${CCCL_SOURCE_DIR}/cub"
    TEST_THRUST_PATH="-I${CCCL_SOURCE_DIR}/thrust"
    TEST_LIBCUDACXX_PATH="-I${CCCL_SOURCE_DIR}/libcudacxx/include"
    TEST_CTK_PATH="-I${CUDA_FIRST_INCLUDE_DIR}"
    TEST_INCLUDE_PATH="${CMAKE_CURRENT_SOURCE_DIR}"
  )

  add_test(NAME ${target_name} COMMAND ${target_name})
endfunction()

file(GLOB test_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  *.cu *.cpp
)

foreach(test_src IN LISTS test_srcs)
  cccl_c_parallel_add_test(test_target "${test_src}")
endforeach()
