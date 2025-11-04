// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nvbench_helper.cuh>

// complex types cannot be compared with operator<
using value_types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
#if NVBENCH_HELPER_HAS_I128
                     int128_t,
#endif
#if _CCCL_HAS_NVFP16() && _CCCL_CTK_AT_LEAST(12, 2)
                     __half,
#endif
#if _CCCL_HAS_NVBF16() && _CCCL_CTK_AT_LEAST(12, 2)
                     __nv_bfloat16,
#endif
                     float,
                     double>;

using op_t = ::cuda::minimum<>;
#include "warp_reduce_base.cuh"
