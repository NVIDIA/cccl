/*
 * Copyright (c) 2025 NVIDIA Corporation
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
#include "testing.cuh" // IWYU pragma: keep
#include <nv/target>

_CCCL_NV_DIAG_SUPPRESS(177) // function "_is_on_device" was declared but never referenced

namespace
{
_CUDAX_API bool _is_on_device() noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, //
                    ({ return false; }),
                    ({ return true; }));
}

void stream_context_test1()
{
  cudax_async::stream_context ctx;
  auto sched = ctx.get_scheduler();

  auto sndr = cudax_async::schedule(sched) //
            | cudax_async::then([] __device__() noexcept -> bool {
                return _is_on_device();
              });

  auto [on_device] = cudax_async::sync_wait(std::move(sndr)).value();
  CHECK(on_device);
}

C2H_TEST("a simple use of the stream context", "[context][stream]")
{
  // put the test in a separate function to avoid an nvc++ issue with
  // extended lambdas in functions with internal linkage (as is the case
  // with C2H tests).
  stream_context_test1();
}
} // namespace
