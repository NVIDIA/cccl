// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#if !TUNE_BASE
#  include <cub/agent/single_pass_scan_operators.cuh>

#  if !defined(TUNE_MAGIC_NS) || !defined(TUNE_L2_WRITE_LATENCY_NS) || !defined(TUNE_DELAY_CONSTRUCTOR_ID)
#    error "TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS, and TUNE_DELAY_CONSTRUCTOR_ID must be defined"
#  endif

using delay_constructor_t =
  cub::detail::delay_constructor_t<TUNE_DELAY_CONSTRUCTOR_ID, TUNE_MAGIC_NS, TUNE_L2_WRITE_LATENCY_NS>;
#endif // !TUNE_BASE
