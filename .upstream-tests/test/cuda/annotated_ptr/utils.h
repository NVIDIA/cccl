//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(_LIBCUDACXX_COMPILER_MSVC)
#pragma warning(disable: 4505)
#endif

#include <cuda/annotated_ptr>

#if defined(DEBUG)
    #define DPRINTF(...) { printf(__VA_ARGS__); }
#else
    #define DPRINTF(...) do {} while (false)
#endif

__device__ __host__
void assert_rt_wrap(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
#ifndef __CUDACC_RTC__
        printf("assert: %s %s %d\n", cudaGetErrorString(code), file, line);
#endif
        assert(code == cudaSuccess);
    }
}
#define assert_rt(ret) { assert_rt_wrap((ret), __FILE__, __LINE__); }

template <typename ... T>
__host__ __device__ constexpr bool unused(T...) {return true;}

template<typename T, int N>
__device__ __host__ __noinline__
T* alloc(bool shared = false) {
  T* arr = nullptr;

#ifdef __CUDA_ARCH__
  if (!shared) {
    arr = (T*)malloc(N * sizeof(T));
  } else {
    __shared__ T data[N];
    arr = data;
  }
#else
  assert_rt(cudaMallocManaged((void**) &arr, N * sizeof(T)));
#endif

  for (int i = 0; i < N; ++i) {
    arr[i] = i;
  }
  return arr;
}

template<typename T>
__device__ __host__ __noinline__
void dealloc(T* arr, bool shared) {
#ifdef __CUDA_ARCH__
  if (!shared) free(arr);
#else
    assert_rt(cudaFree(arr));
#endif
}
