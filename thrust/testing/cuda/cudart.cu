#include <cuda/std/memory>

#include <cuda_runtime_api.h>

#include <unittest/unittest.h>

template <typename T>
void test_cuda_malloc_result_aligned(const std::size_t n)
{
  T* ptr = nullptr;
  cudaMalloc(&ptr, n * sizeof(T));
  cudaFree(ptr);

  REQUIRE(::cuda::std::is_sufficiently_aligned<alignof(T)>(ptr));
}
DECLARE_VARIABLE_UNITTEST(test_cuda_malloc_result_aligned);
