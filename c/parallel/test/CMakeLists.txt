add_executable(cccl.c.parallel.test.reduce test_reduce.cpp test_main.cpp)
cccl_configure_target(cccl.c.parallel.test.reduce DIALECT 20)

target_link_libraries(cccl.c.parallel.test.reduce PRIVATE cccl.c CUDA::cudart CUDA::nvrtc Catch2::Catch2)

target_compile_definitions(cccl.c.parallel.test.reduce PRIVATE CCCL_C_EXPERIMENTAL
                                                     TEST_CUB_PATH="-I${CCCL_SOURCE_DIR}/cub"
                                                     TEST_THRUST_PATH="-I${CCCL_SOURCE_DIR}/cub"
                                                     TEST_LIBCUDACXX_PATH="-I${CCCL_SOURCE_DIR}/libcudacxx/include"
                                                     TEST_CTK_PATH="-I${CUDAToolkit_INCLUDE_DIRS}")

add_test(NAME cccl.c.parallel.test.reduce COMMAND cccl.c.parallel.test.reduce)
