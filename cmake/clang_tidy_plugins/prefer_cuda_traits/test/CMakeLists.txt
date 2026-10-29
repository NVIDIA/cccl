list(APPEND CMAKE_MESSAGE_CONTEXT "test")

file(GLOB test_srcs CONFIGURE_DEPENDS "*.cpp")

foreach (src IN LISTS test_srcs)
  cccl_create_clang_tidy_test(prefer_cuda_traits "${src}")
endforeach()
