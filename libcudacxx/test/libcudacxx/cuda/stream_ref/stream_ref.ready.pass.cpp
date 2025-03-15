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

#include "test_macros.h"

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

int main(int argc, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST,
               ( // passing case
                 cudaStream_t stream; cudaStreamCreate(&stream); cuda::stream_ref ref{stream}; test_ready(ref);
                 cudaStreamDestroy(stream);))

  return 0;
}
