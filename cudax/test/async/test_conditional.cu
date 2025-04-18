/*
 * Copyright (c) 2024 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Include this first
#include <cuda/experimental/__async/sender.cuh>

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

    CUDAX_CHECK(even == (i % 2 == 0));
    CUDAX_CHECK(odd == (i % 2 == 1));
  }
}

} // namespace
