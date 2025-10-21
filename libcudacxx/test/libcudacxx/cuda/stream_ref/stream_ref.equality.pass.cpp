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
#include <cuda/stream>

__host__ __device__ void test()
{
  cuda::stream_ref left{reinterpret_cast<cudaStream_t>(42)};
  cuda::stream_ref right{reinterpret_cast<cudaStream_t>(1337)};
  static_assert(noexcept(left == right), "");
  static_assert(noexcept(left != right), "");

  assert(left == left);
  assert(left != right);
}

int main(int argc, char** argv)
{
  test();

  return 0;
}
