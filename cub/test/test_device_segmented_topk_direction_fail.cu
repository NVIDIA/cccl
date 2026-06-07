// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile-fail test that locks in the compile-time-only selection-direction contract of the batched top-k: the
// direction must be a compile-time `::cuda::__argument::__constant<Dir>`. Passing a runtime `cub::detail::topk::select`
// (or, by the same overload, a per-segment iterator of directions) must fail to compile. Reintroducing an overload
// that accepts a runtime direction would make this test compile and therefore fail.

#include <cub/device/dispatch/dispatch_batched_topk.cuh>

int main()
{
  // A runtime selection direction is not a compile-time `__constant<Dir>` and must be rejected.
  [[maybe_unused]] auto wrapped = cub::detail::batched_topk::wrap_select_direction(cub::detail::topk::select::max);
  // expected-error {{selection direction must be a compile-time option}}
}
