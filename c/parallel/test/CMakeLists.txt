cccl_get_catch2()

function(cccl_c_parallel_add_test target_name_var source)
  string(REGEX REPLACE "test_([^.]*)" "cccl.c.parallel.test.\\1" target_name "${source}")
  set(target_name_var ${target_name} PARENT_SCOPE)

  add_executable(${target_name}
    "${source}"
    test_main.cpp
  )
  cccl_configure_target(${target_name} DIALECT 20)

  target_link_libraries(${target_name} PRIVATE
    cccl.c.parallel
    CUDA::cudart
    CUDA::nvrtc
    Catch2::Catch2
    cccl.compiler_interface_cpp20
  )

  target_compile_definitions(${target_name} PRIVATE
    TEST_CUB_PATH="-I${CCCL_SOURCE_DIR}/cub"
    TEST_THRUST_PATH="-I${CCCL_SOURCE_DIR}/thrust"
    TEST_LIBCUDACXX_PATH="-I${CCCL_SOURCE_DIR}/libcudacxx/include"
    TEST_CTK_PATH="-I${CUDAToolkit_INCLUDE_DIRS}"
  )

  add_test(NAME ${target_name} COMMAND ${target_name})
endfunction()

file(GLOB test_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  *.cu *.cpp
)
list(REMOVE_ITEM test_srcs test_main.cpp)

foreach(test_src IN LISTS test_srcs)
  cccl_c_parallel_add_test(test_target "${test_src}")
endforeach()
