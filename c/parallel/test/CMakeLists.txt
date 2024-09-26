add_executable(cccl.c.test.reduce test_reduce.cpp test_main.cpp)

target_link_libraries(cccl.c.test.reduce PRIVATE cccl.c CUDA::cudart CUDA::nvrtc Catch2::Catch2)

target_compile_definitions(cccl.c.test.reduce PRIVATE CCCL_C_EXPERIMENTAL
                                                     TEST_CUB_PATH="-I${CCCL_SOURCE_DIR}/cub"
                                                     TEST_THRUST_PATH="-I${CCCL_SOURCE_DIR}/cub"
                                                     TEST_LIBCUDACXX_PATH="-I${CCCL_SOURCE_DIR}/libcudacxx/include"
                                                     TEST_CTK_PATH="-I${CUDAToolkit_INCLUDE_DIRS}")
