// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#if !TUNE_BASE
#  include <cub/detail/delay_constructor.cuh>

#  include <nvbench_helper.cuh>

#  if !defined(TUNE_MAGIC_NS) || !defined(TUNE_L2_WRITE_LATENCY_NS) || !defined(TUNE_DELAY_CONSTRUCTOR_ID)
#    error "TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS, and TUNE_DELAY_CONSTRUCTOR_ID must be defined"
#  endif

using delay_constructor_t =
  cub::detail::delay_constructor_t<static_cast<cub::detail::delay_constructor_kind>(TUNE_DELAY_CONSTRUCTOR_ID),
                                   TUNE_MAGIC_NS,
                                   TUNE_L2_WRITE_LATENCY_NS>;

inline constexpr auto delay_constructor_policy = cub::detail::delay_constructor_policy{
  static_cast<cub::detail::delay_constructor_kind>(TUNE_DELAY_CONSTRUCTOR_ID), TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS};

#endif // !TUNE_BASE
