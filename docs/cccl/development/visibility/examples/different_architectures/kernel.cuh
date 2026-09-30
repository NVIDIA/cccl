#pragma once

#include <cstdint>
#include <cstdio>

template <int... Archs>
__attribute__((visibility("hidden"))) __host__ __device__ constexpr int sum_archs()
{
  return (Archs + ... + 0);
}

template <class T, auto Archs = sum_archs<__CUDA_ARCH_LIST__>()>
__attribute__((visibility("hidden"))) __global__ void kernel(char ln, T* val)
{
  printf("%c: kernel: set val = %i\n", ln, sum_archs<__CUDA_ARCH_LIST__>());
  *val = sum_archs<__CUDA_ARCH_LIST__>();
}

__attribute__((visibility("hidden"))) __forceinline__ int use_kernel()
{
  int* d_val{};
  cudaMalloc(&d_val, sizeof(size_t));
  kernel<<<1, 1>>>(d_val);
  int ret;
  if (cudaMemcpy(&ret, d_val, sizeof(size_t), cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    printf("c: FAILED to copy from device to host\n");
  }
  return ret;
}

template <class T = int>
struct some_class_with_kernel
{
  T val_;

  some_class_with_kernel();
  __forceinline__ some_class_with_kernel(T)
  {
    val_ = use_kernel();
  }
};
