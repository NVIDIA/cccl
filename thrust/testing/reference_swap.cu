// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/memory.h>

#include <cuda/std/__algorithm/iter_swap.h>
#include <cuda/std/__utility/swap.h>

#include <catch2/catch_test_macros.hpp>

// a pointer with the default reference type, i.e. thrust::reference<int, pointer<...>>
using pointer_type   = thrust::pointer<int, thrust::host_system_tag>;
using reference_type = pointer_type::reference;

TEST_CASE("ThrustReferenceIsSwappable", "[reference]")
{
  STATIC_REQUIRE(::cuda::std::is_swappable_v<reference_type>);
}

TEST_CASE("ThrustReferenceSwap", "[reference]")
{
  int data[] = {1, 2};
  const pointer_type p{data};

  using ::cuda::std::swap;
  swap(*p, *(p + 1)); // proxy rvalues, found via ADL
  CHECK(data[0] == 2);
  CHECK(data[1] == 1);
}

TEST_CASE("ThrustReferenceIterSwap", "[reference]")
{
  int data[] = {1, 2};
  const pointer_type p{data};

  ::cuda::std::iter_swap(p, p + 1);
  CHECK(data[0] == 2);
  CHECK(data[1] == 1);
}
