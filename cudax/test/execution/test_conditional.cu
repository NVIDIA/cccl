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

namespace
{
C2H_TEST("simple use of conditional runs exactly one of the two closures", "[adaptors][conditional]")
{
  for (int i = 42; i < 44; ++i)
  {
    bool even{false};
    bool odd{false};

    auto sndr1 =
      cudax_async::just(i)
      | cudax_async::conditional(
        [](int i) {
          return i % 2 == 0;
        },
        cudax_async::then([&](int) {
          even = true;
        }),
        cudax_async::then([&](int) {
          odd = true;
        }));

    check_value_types<types<>>(sndr1);
    check_sends_stopped<false>(sndr1);
#if _CCCL_HAS_EXCEPTIONS()
    check_error_types<std::exception_ptr>(sndr1);
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
    check_error_types<>(sndr1);
#endif // !_CCCL_HAS_EXCEPTIONS()

    auto op = cudax_async::connect(std::move(sndr1), checked_value_receiver<>{});
    cudax_async::start(op);

    CHECK(even == (i % 2 == 0));
    CHECK(odd == (i % 2 == 1));
  }
}

} // namespace
