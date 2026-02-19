// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#include <cub/detail/warpspeed/squad/squad_desc.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
struct scan_warpspeed_policy
{
  static constexpr int num_squads = 5;

  bool valid = false;
  int num_reduce_warps;
  int num_scan_stor_warps;
  int num_load_warps;
  int num_sched_warps;
  int num_look_ahead_warps;

  int num_look_ahead_items;
  int num_total_threads;
  int items_per_thread;
  int tile_size;

  _CCCL_API constexpr explicit operator bool() const noexcept
  {
    return valid;
  }

  _CCCL_API constexpr warpspeed::SquadDesc squadReduce() const
  {
    return warpspeed::SquadDesc{0, num_reduce_warps};
  }
  _CCCL_API constexpr warpspeed::SquadDesc squadScanStore() const
  {
    return warpspeed::SquadDesc{1, num_scan_stor_warps};
  }
  _CCCL_API constexpr warpspeed::SquadDesc squadLoad() const
  {
    return warpspeed::SquadDesc{2, num_load_warps};
  }
  _CCCL_API constexpr warpspeed::SquadDesc squadSched() const
  {
    return warpspeed::SquadDesc{3, num_sched_warps};
  }
  _CCCL_API constexpr warpspeed::SquadDesc squadLookback() const
  {
    return warpspeed::SquadDesc{4, num_look_ahead_warps};
  }

  _CCCL_API constexpr friend bool operator==(const scan_warpspeed_policy& lhs, const scan_warpspeed_policy& rhs)
  {
    return lhs.valid == rhs.valid && lhs.num_reduce_warps == rhs.num_reduce_warps
        && lhs.num_scan_stor_warps == rhs.num_scan_stor_warps && lhs.num_load_warps == rhs.num_load_warps
        && lhs.num_sched_warps == rhs.num_sched_warps && lhs.num_look_ahead_warps == rhs.num_look_ahead_warps
        && lhs.num_look_ahead_items == rhs.num_look_ahead_items && lhs.num_total_threads == rhs.num_total_threads
        && lhs.items_per_thread == rhs.items_per_thread && lhs.tile_size == rhs.tile_size;
  }
};
} // namespace detail::scan

CUB_NAMESPACE_END
