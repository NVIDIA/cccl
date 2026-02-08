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
#include "common/checked_receiver.cuh"
#include "common/utility.cuh"
#include "testing.cuh"

namespace ex = cuda::experimental::execution;

namespace
{
C2H_TEST("simple use of conditional runs exactly one of the two closures", "[adaptors][conditional]")
{
  for (int i = 42; i < 44; ++i)
  {
    bool even{false};
    bool odd{false};

    auto sndr1 =
      ex::just(i)
      | ex::conditional(
        [](int i) {
          return i % 2 == 0;
        },
        ex::then([&](int) {
          even = true;
        }),
        ex::then([&](int) {
          odd = true;
        }));

    check_value_types<types<>>(sndr1);
    check_sends_stopped<false>(sndr1);
#if _CCCL_HAS_EXCEPTIONS()
    check_error_types<ex::exception_ptr>(sndr1);
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
    check_error_types<>(sndr1);
#endif // !_CCCL_HAS_EXCEPTIONS()

    auto op = ex::connect(std::move(sndr1), checked_value_receiver<>{});
    ex::start(op);

    CHECK(even == (i % 2 == 0));
    CHECK(odd == (i % 2 == 1));
  }
}
} // namespace
