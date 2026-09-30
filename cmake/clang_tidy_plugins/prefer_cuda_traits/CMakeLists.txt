list(APPEND CMAKE_MESSAGE_CONTEXT "prefer_cuda_traits")

cccl_create_clang_tidy_plugin(prefer_cuda_traits SOURCES prefer_cuda_traits.cpp)

add_subdirectory(test)
