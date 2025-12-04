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

#include <cuda/memory_resource>
#include <cuda/std/__pstl/for_each.h>
#include <cuda/std/execution>
#include <cuda/std/memory>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <nv/target>

template <class Policy>
void test(Policy pol)
{
  namespace execution = cuda::std::execution;
  static_assert(!execution::__queryable_with<execution::sequenced_policy, ::cuda::get_stream_t>);
  static_assert(!execution::__queryable_with<execution::parallel_policy, ::cuda::get_stream_t>);
  static_assert(!execution::__queryable_with<execution::parallel_unsequenced_policy, ::cuda::get_stream_t>);
  static_assert(!execution::__queryable_with<execution::unsequenced_policy, ::cuda::get_stream_t>);

  { // Ensure that the plain policy returns a well defined stream
    cuda::stream_ref expected_stream{cudaStreamPerThread};
    assert(cuda::get_stream(pol) == expected_stream);
  }

  { // Ensure that we can attach a stream to an execution policy
    cuda::stream stream{cuda::device_ref{0}};
    auto pol_with_stream = pol.set_stream(stream);
    assert(cuda::get_stream(pol_with_stream) == stream);

    using stream_policy_t = decltype(pol_with_stream);
    static_assert(noexcept(pol.set_stream(stream)));
    static_assert(cuda::std::is_execution_policy_v<stream_policy_t>);
    static_assert(cuda::std::is_base_of_v<cuda::std::execution::__policy_stream_holder<true>, stream_policy_t>);
  }

  { // Ensure that attaching a stream multiple times just overwrites the old stream
    cuda::stream stream{cuda::device_ref{0}};
    auto pol_with_stream = pol.set_stream(stream);
    assert(cuda::get_stream(pol_with_stream) == stream);

    using stream_policy_t = decltype(pol_with_stream);
    cuda::stream other_stream{cuda::device_ref{0}};
    decltype(auto) pol_with_other_stream = pol_with_stream.set_stream(other_stream);
    static_assert(cuda::std::is_same_v<decltype(pol_with_other_stream), stream_policy_t&>);
    assert(::cuda::get_stream(pol_with_stream) == other_stream);
    assert(::cuda::get_stream(pol_with_other_stream) == other_stream);
    assert(cuda::std::addressof(pol_with_stream) == cuda::std::addressof(pol_with_other_stream));
  }
}

void test()
{
  namespace execution = cuda::std::execution;
  static_assert(!execution::__queryable_with<execution::sequenced_policy, ::cuda::get_stream_t>);
  static_assert(!execution::__queryable_with<execution::parallel_policy, ::cuda::get_stream_t>);
  static_assert(!execution::__queryable_with<execution::parallel_unsequenced_policy, ::cuda::get_stream_t>);
  static_assert(!execution::__queryable_with<execution::unsequenced_policy, ::cuda::get_stream_t>);

  test(cuda::execution::__cub_par_unseq);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))

  return 0;
}
