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

#define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#include <cuda/stream_ref>
#include <cuda/std/cassert>
#include <atomic>
#include <chrono>
#include <thread>

void CUDART_CB callback(cudaStream_t, cudaError_t, void* flag)
{
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(1000ms);
  assert(!reinterpret_cast<std::atomic_flag*>(flag)->test_and_set());
}

void test_wait(cuda::stream_ref& ref) {
  #ifndef _LIBCUDACXX_NO_EXCEPTIONS
      try {
        ref.wait();
      } catch (...) {
        assert(false && "Should not have thrown");
      }
  #else
      ref.wait();
  #endif // _LIBCUDACXX_NO_EXCEPTIONS
}

int main(int argc, char** argv) {
    NV_IF_TARGET(NV_IS_HOST,( // passing case
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        std::atomic_flag flag(ATOMIC_FLAG_INIT);
        cudaStreamAddCallback(stream, callback, &flag, 0);
        cuda::stream_ref ref{stream};
        test_wait(ref);
        assert(flag.test_and_set());
        cudaStreamDestroy(stream);
    ))

    return 0;
}
