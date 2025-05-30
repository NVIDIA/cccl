//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Include this first
#include <cuda/experimental/execution.cuh>

// Then include the test helpers
#include <nv/target>

#include "common/checked_receiver.cuh"
#include "common/utility.cuh"
#include "testing.cuh"

namespace
{
C2H_TEST("simple use of sequence executes both child operations", "[adaptors][sequence]")
{
  bool flag1{false};
  bool flag2{false};

  auto sndr1 = cudax_async::sequence(
    cudax_async::just() | cudax_async::then([&] {
      flag1 = true;
    }),
    cudax_async::just() | cudax_async::then([&] {
      flag2 = true;
    }));

  check_value_types<types<>>(sndr1);
  check_sends_stopped<false>(sndr1);
  NV_IF_ELSE_TARGET(NV_IS_HOST, //
                    ({ check_error_types<std::exception_ptr>(sndr1); }),
                    ({ check_error_types<>(sndr1); }));

  auto op = cudax_async::connect(std::move(sndr1), checked_value_receiver<>{});
  cudax_async::start(op);

  CHECK(flag1);
  CHECK(flag2);
}

} // namespace
