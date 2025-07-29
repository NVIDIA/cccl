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

__host__ __device__ void test()
{
  static_assert(cuda::std::execution::__queryable_with<cuda::stream_ref, cuda::get_stream_t>);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(42);
  cuda::stream_ref ref{stream};
  assert(ref == ref.query(cuda::get_stream));
}

int main(int argc, char** argv)
{
  test();

  return 0;
}
