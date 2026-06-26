//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/functional>

#include <cuda/experimental/__multi_gpu/concepts.h>
#include <cuda/experimental/__multi_gpu/nccl_communicator_ref.h>

#include <functional>

#include <testing.cuh>

namespace
{
struct payload
{
  int from;
  int value;
};

struct non_trivial
{
  non_trivial(const non_trivial&) {} // NOLINT(modernize-use-equals-default)

  int value;
};

struct unsupported_op
{};
} // namespace

C2H_TEST("nccl_communicator_ref concept conformance", "[multi_gpu][nccl]")
{
  STATIC_REQUIRE(cudax::__communicator<cudax::nccl_communicator_ref>);
  STATIC_REQUIRE(cudax::__has_send<cudax::nccl_communicator_ref, int*>);
  STATIC_REQUIRE(cudax::__has_send<cudax::nccl_communicator_ref, payload*>);
  STATIC_REQUIRE(!cudax::__has_send<cudax::nccl_communicator_ref, non_trivial*>);
  STATIC_REQUIRE(cudax::__has_recv<cudax::nccl_communicator_ref, int*>);
  STATIC_REQUIRE(cudax::__has_recv<cudax::nccl_communicator_ref, payload*>);
  STATIC_REQUIRE(!cudax::__has_recv<cudax::nccl_communicator_ref, non_trivial*>);
  STATIC_REQUIRE(cudax::__has_reduce<cudax::nccl_communicator_ref, int*>);
  STATIC_REQUIRE(!cudax::__has_reduce<cudax::nccl_communicator_ref, payload*>);
  STATIC_REQUIRE(cudax::__has_all_reduce<cudax::nccl_communicator_ref>);
  STATIC_REQUIRE(!cudax::__has_all_reduce<cudax::nccl_communicator_ref, payload*>);
  STATIC_REQUIRE(cudax::__has_gather<cudax::nccl_communicator_ref>);
  STATIC_REQUIRE(cudax::__has_gather<cudax::nccl_communicator_ref, payload*>);
  STATIC_REQUIRE(!cudax::__has_gather<cudax::nccl_communicator_ref, non_trivial*>);
  STATIC_REQUIRE(cudax::__has_gather_v<cudax::nccl_communicator_ref>);
  STATIC_REQUIRE(cudax::__has_gather_v<cudax::nccl_communicator_ref, payload*>);
  STATIC_REQUIRE(!cudax::__has_gather_v<cudax::nccl_communicator_ref, non_trivial*>);
  STATIC_REQUIRE(cudax::__has_all_gather<cudax::nccl_communicator_ref>);
  STATIC_REQUIRE(cudax::__has_all_gather<cudax::nccl_communicator_ref, payload*>);
  STATIC_REQUIRE(!cudax::__has_all_gather<cudax::nccl_communicator_ref, non_trivial*>);
  STATIC_REQUIRE(cudax::__has_broadcast<cudax::nccl_communicator_ref>);
  STATIC_REQUIRE(cudax::__has_broadcast<cudax::nccl_communicator_ref, payload*>);
  STATIC_REQUIRE(!cudax::__has_broadcast<cudax::nccl_communicator_ref, non_trivial*>);
  STATIC_REQUIRE(cudax::__has_all_to_all<cudax::nccl_communicator_ref>);
  STATIC_REQUIRE(cudax::__has_all_to_all<cudax::nccl_communicator_ref, payload*>);
  STATIC_REQUIRE(!cudax::__has_all_to_all<cudax::nccl_communicator_ref, non_trivial*>);
  STATIC_REQUIRE(cudax::__has_all_to_all_v<cudax::nccl_communicator_ref>);
  STATIC_REQUIRE(cudax::__has_all_to_all_v<cudax::nccl_communicator_ref, payload*>);
  STATIC_REQUIRE(!cudax::__has_all_to_all_v<cudax::nccl_communicator_ref, non_trivial*>);

  STATIC_REQUIRE(cudax::nccl_transportable<int>);
  STATIC_REQUIRE(cudax::nccl_transportable<int*>);
  STATIC_REQUIRE(cudax::nccl_transportable<const int*>);
  STATIC_REQUIRE(cudax::nccl_transportable<const volatile int* const>);
  STATIC_REQUIRE(cudax::nccl_transportable<float>);
  STATIC_REQUIRE(cudax::nccl_transportable<::cuda::std::int32_t>);
  STATIC_REQUIRE(cudax::nccl_transportable<void>);
  STATIC_REQUIRE(cudax::nccl_transportable<payload>);
  STATIC_REQUIRE(!cudax::nccl_transportable<non_trivial>);

  STATIC_REQUIRE(cudax::nccl_reducible<int, cuda::std::plus<>>);
  STATIC_REQUIRE(cudax::nccl_reducible<int, cuda::std::multiplies<>>);
  STATIC_REQUIRE(cudax::nccl_reducible<int, cuda::maximum<>>);
  STATIC_REQUIRE(cudax::nccl_reducible<int, cuda::minimum<>>);
  STATIC_REQUIRE(!cudax::nccl_reducible<payload, cuda::std::plus<>>);
  STATIC_REQUIRE(!cudax::nccl_reducible<int, unsupported_op>);

  STATIC_REQUIRE(cudax::nccl_reducible<int, std::plus<>>);
  STATIC_REQUIRE(cudax::nccl_reducible<int, std::multiplies<>>);
  STATIC_REQUIRE(!cudax::nccl_reducible<payload, std::plus<>>);
}
