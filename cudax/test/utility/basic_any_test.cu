//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__utility/basic_any.cuh>

#include <testing.cuh>

#undef interface

template <class...>
struct iempty : cudax::interface<iempty>
{};

void TESTTEST()
{
  cudax::basic_any<iempty<>> a{42};
  CHECK(a.has_value() == true);
  CHECK(a.type() == _CCCL_TYPEID(int));
  CHECK(a.interface() == _CCCL_TYPEID(iempty<>));
  REQUIRE(cudax::any_cast<int>(&a));
  CHECK(cudax::any_cast<int>(&a) == cudax::any_cast<void>(&a));
  CHECK(*cudax::any_cast<int>(&a) == 42);
}

TEST_CASE("a basic_any test", "[utility][basic_any]")
{
  NV_IF_TARGET(NV_IS_HOST, (TESTTEST();))
}
