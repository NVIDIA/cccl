//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/atomic>

#include <cuda/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

/*
Test goals:
Interleaved 8b/16b access to a 32b window while there is thread contention.

for 8b:
Launch 1020 threads, fetch_add(1) each window, value at end of kernel should be 0xFF..FF. This checks for corruption
caused by interleaved access to different parts of the window.

for 16b:
Launch 1020 threads, fetch_add(128) into both windows.
*/

template <class T, int Inc>
__host__ __device__ void fetch_add_into_window(T* window, uint16_t* atomHistory)
{
  typedef cuda::atomic_ref<T, cuda::thread_scope_block> Atom;

  Atom a(*window);
  *atomHistory = a.fetch_add(Inc);
}

template <class T, int Inc>
__device__ void device_do_test(uint32_t expected)
{
  __shared__ uint16_t atomHistory[1024];
  __shared__ uint32_t atomicStorage;
  cuda::atomic_ref<uint32_t, cuda::thread_scope_block> bucket(atomicStorage);

  constexpr uint32_t offsetMask = ((4 / sizeof(T)) - 1);
  // Access offset is interleaved meaning threads 4, 5, 6, 7 access window 0, 1, 2, 3 and so on.
  const uint32_t threadOffset = threadIdx.x & offsetMask;

  if (threadIdx.x == 0)
  {
    bucket.store(0);
  }
  __syncthreads();

  T* window = reinterpret_cast<T*>(&atomicStorage) + threadOffset;
  fetch_add_into_window<T, Inc>(window, atomHistory + threadIdx.x);

  __syncthreads();
  if (threadIdx.x == 0)
  {
    printf("expected: 0x%X\r\n", expected);
    printf("result:   0x%X\r\n", bucket.load());
    assert(bucket.load() == expected);
  }
}

int main(int, char**)
{
  NV_DISPATCH_TARGET(NV_IS_HOST,
                     (cuda_thread_count = 1020;),
                     NV_IS_DEVICE,
                     (device_do_test<uint8_t, 1>(~uint32_t(0)); device_do_test<uint16_t, 128>(0xFF00FF00);));

  return 0;
}
