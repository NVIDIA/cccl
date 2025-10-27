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

#include <cuda/std/execution>
#include <cuda/std/type_traits>
#include <cuda/stream>

template <class Policy>
void test(Policy pol)
{
  { // Ensure that the plain policy returns a well defined stream
    cuda::stream_ref expected_stream{cudaStreamPerThread};
    assert(cuda::get_stream(pol) == expected_stream);
  }

  { // Ensure that we can attach a stream to an execution policy
    cuda::stream stream{cuda::device_ref{0}};
    auto pol_with_stream = pol.set_stream(stream);
    assert(cuda::get_stream(pol_with_stream) == stream);

    static_assert(noexcept(pol.set_stream(stream)));
    static_assert(cuda::std::is_base_of_v<Policy, decltype(pol_with_stream)>);
    static_assert(cuda::std::is_execution_policy_v<decltype(pol_with_stream)>);
  }
}

void test()
{
  test(cuda::std::execution::seq);
  test(cuda::std::execution::par);
  test(cuda::std::execution::unseq);
  test(cuda::std::execution::par_unseq);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))

  return 0;
}
