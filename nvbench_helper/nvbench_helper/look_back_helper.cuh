// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#if !TUNE_BASE
#  include <cub/agent/single_pass_scan_operators.cuh>

#  include <nvbench_helper.cuh>

#  if !defined(TUNE_MAGIC_NS) || !defined(TUNE_L2_WRITE_LATENCY_NS) || !defined(TUNE_DELAY_CONSTRUCTOR_ID)
#    error "TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS, and TUNE_DELAY_CONSTRUCTOR_ID must be defined"
#  endif

using delay_constructors = nvbench::type_list<
  cub::detail::no_delay_constructor_t<TUNE_L2_WRITE_LATENCY_NS>,
  cub::detail::fixed_delay_constructor_t<TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS>,
  cub::detail::exponential_backoff_constructor_t<TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS>,
  cub::detail::exponential_backoff_jitter_constructor_t<TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS>,
  cub::detail::exponential_backoff_jitter_window_constructor_t<TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS>,
  cub::detail::exponential_backon_jitter_window_constructor_t<TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS>,
  cub::detail::exponential_backon_jitter_constructor_t<TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS>,
  cub::detail::exponential_backon_constructor_t<TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS>>;

using delay_constructor_t = nvbench::tl::get<TUNE_DELAY_CONSTRUCTOR_ID, delay_constructors>;
#endif // !TUNE_BASE
