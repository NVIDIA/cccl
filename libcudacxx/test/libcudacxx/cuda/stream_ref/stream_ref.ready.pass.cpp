//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/std/cassert>
#include <cuda/stream>

#include "test_macros.h"

_CCCL_SUPPRESS_DEPRECATED_PUSH

void test_ready(cuda::stream_ref& ref)
{
#if TEST_HAS_EXCEPTIONS()
  try
  {
    assert(ref.ready());
  }
  catch (...)
  {
    assert(false && "Should not have thrown");
  }
#else
  assert(ref.ready());
#endif // TEST_HAS_EXCEPTIONS()
}

bool test()
{
  cudaStream_t stream;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  cuda::stream_ref ref{stream};
  test_ready(ref);
  assert(cudaStreamDestroy(stream) == cudaSuccess);

  return true;
}

int main(int argc, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
