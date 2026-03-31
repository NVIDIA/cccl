// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
struct scan_warpspeed_policy
{
  bool valid = false;
  int num_reduce_and_scan_warps;
  int look_ahead_items_per_thread;
  int items_per_thread;

  _CCCL_API constexpr explicit operator bool() const noexcept
  {
    return valid;
  }

  _CCCL_API constexpr int tile_size() const noexcept
  {
    return items_per_thread * num_reduce_and_scan_warps * warp_threads;
  }

  _CCCL_API constexpr friend bool operator==(const scan_warpspeed_policy& lhs, const scan_warpspeed_policy& rhs)
  {
    return lhs.valid == rhs.valid && lhs.num_reduce_and_scan_warps == rhs.num_reduce_and_scan_warps
        && lhs.look_ahead_items_per_thread == rhs.look_ahead_items_per_thread
        && lhs.items_per_thread == rhs.items_per_thread;
  }

  _CCCL_API constexpr friend bool operator!=(const scan_warpspeed_policy& lhs, const scan_warpspeed_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const scan_warpspeed_policy& p)
  {
    return os << "scan_warpspeed_policy { .valid = " << p.valid << ", .num_reduce_and_scan_warps = "
              << p.num_reduce_and_scan_warps << ", .look_ahead_items_per_thread = " << p.look_ahead_items_per_thread
              << ", .items_per_thread = " << p.items_per_thread << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};
} // namespace detail::scan

CUB_NAMESPACE_END
