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

int main(int argc, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST,
               (cudaStream_t stream = reinterpret_cast<cudaStream_t>(42); cuda::stream_ref ref{stream};
                assert(ref.get() == stream);))

  return 0;
}
