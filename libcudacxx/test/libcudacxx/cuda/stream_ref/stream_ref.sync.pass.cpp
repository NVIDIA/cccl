//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/std/cassert>
#include <cuda/stream_ref>

#include <atomic>
#include <chrono>
#include <thread>

#include "test_macros.h"

void CUDART_CB callback(cudaStream_t, cudaError_t, void* flag)
{
  std::chrono::milliseconds sleep_duration{1000};
  std::this_thread::sleep_for(sleep_duration);
  assert(!reinterpret_cast<std::atomic_flag*>(flag)->test_and_set());
}

void test_sync(cuda::stream_ref& ref)
{
#if TEST_HAS_EXCEPTIONS()
  try
  {
    ref.sync();
  }
  catch (...)
  {
    assert(false && "Should not have thrown");
  }
#else
  ref.sync();
#endif // TEST_HAS_EXCEPTIONS()
}

int main(int argc, char** argv)
{
  NV_IF_TARGET(
    NV_IS_HOST,
    ( // passing case
      cudaStream_t stream; cudaStreamCreate(&stream); std::atomic_flag flag = ATOMIC_FLAG_INIT;
      cudaStreamAddCallback(stream, callback, &flag, 0);
      cuda::stream_ref ref{stream};
      test_sync(ref);
      assert(flag.test_and_set());
      cudaStreamDestroy(stream);))

  return 0;
}
