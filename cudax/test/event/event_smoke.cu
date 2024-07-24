//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/event.cuh>

#include "../hierarchy/testing_common.cuh"
#include <catch2/catch.hpp>

namespace
{
cudax::event_ref fn_takes_event_ref(cudax::event_ref ref)
{
  return ref;
}
} // namespace

TEST_CASE("can construct an event_ref from a cudaEvent_t", "[event]")
{
  ::cudaEvent_t event;
  CUDAX_REQUIRE(::cudaEventCreate(&event) == ::cudaSuccess);
  cudax::event_ref ref(event);
  CUDAX_REQUIRE(ref.get() == event);
  // test implicit converstion from cudaEvent_t:
  cudax::event_ref ref2 = ::fn_takes_event_ref(event);
  CUDAX_REQUIRE(ref2.get() == event);
  CUDAX_REQUIRE(::cudaEventDestroy(event) == ::cudaSuccess);
}

TEST_CASE("can copy construct an event_ref and compare for equality", "[event]")
{
  ::cudaEvent_t event;
  CUDAX_REQUIRE(::cudaEventCreate(&event) == ::cudaSuccess);
  const cudax::event_ref ref(event);
  const cudax::event_ref ref2 = ref;
  CUDAX_REQUIRE(ref2 == ref);
  CUDAX_REQUIRE(!(ref != ref2));
  CUDAX_REQUIRE((ref ? true : false)); // test contextual convertibility to bool
  CUDAX_REQUIRE(!!ref);
  CUDAX_REQUIRE(::cudaEvent_t{} != ref);
  CUDAX_REQUIRE(::cudaEventDestroy(event) == ::cudaSuccess);
}
