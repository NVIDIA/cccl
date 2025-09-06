// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <thrust/logical.h>

#include <c2h/catch2_test_helper.h>

C2H_TEST("Tests c2h::gen for cudax::async_buffer", "[c2h]")
{
  namespace cudax = cuda::experimental;

  auto dev    = cuda::device_ref{0};
  auto stream = cudax::stream{dev};
  auto env    = cudax::env_t<cuda::mr::device_accessible>{cudax::device_memory_resource{dev}, stream};
  auto buf    = cudax::async_device_buffer<int>{env, 1000, cudax::no_init};

  SECTION("seed")
  {
    c2h::gen(C2H_SEED(1), buf, 10, 20);

    using thrust::placeholders::_1;
    const auto r = thrust::all_of(c2h::device_policy, buf.begin(), buf.end(), 10 <= _1 && _1 <= 20);
    REQUIRE(r);
  }
  SECTION("mod")
  {
    c2h::gen(c2h::modulo_t{3}, buf);

    using thrust::placeholders::_1;
    const auto r = thrust::all_of(c2h::device_policy, buf.begin(), buf.end(), 0 <= _1 && _1 < 3);
    REQUIRE(r);
  }
}
