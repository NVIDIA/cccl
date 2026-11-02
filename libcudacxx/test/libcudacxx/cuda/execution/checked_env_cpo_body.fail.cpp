// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/execution>
#include <cuda/stream_ref>

struct throwing_env
{
  cuda::stream_ref query(cuda::get_stream_t) const
  {
    return cuda::stream_ref{cudaStream_t{}};
  }
};

int main(int, char**)
{
  // expected-error@* {{static assertion failed}}
  auto env = cuda::checked_env<cuda::get_stream_t>(throwing_env{});
  (void) env;
  return 0;
}
