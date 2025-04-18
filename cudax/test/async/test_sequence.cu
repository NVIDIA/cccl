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
#include <nv/target>

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

  CUDAX_CHECK(flag1);
  CUDAX_CHECK(flag2);
}

} // namespace
