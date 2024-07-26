//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/stream.cuh>

#include "../common/utility.cuh"
#include "../hierarchy/testing_common.cuh"
#include <catch2/catch.hpp>

TEST_CASE("Stream create", "[stream]")
{
  cudax::stream str1, str2;
  cudax::event ev1(str1);

  auto ev2 = str1.record_event();
  str2.wait(str1);
}
