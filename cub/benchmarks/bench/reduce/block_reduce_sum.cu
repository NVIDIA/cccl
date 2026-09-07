// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <nvbench_helper.cuh>

using value_types =
  push_back_t<all_types
#if _CCCL_HAS_NVFP16() && _CCCL_CTK_AT_LEAST(12, 2)
              ,
              __half
#endif
#if _CCCL_HAS_NVBF16() && _CCCL_CTK_AT_LEAST(12, 2)
              ,
              __nv_bfloat16
#endif
              >;

using op_t = ::cuda::std::plus<>;
#include "block_reduce_base.cuh"
