// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nvbench_helper.cuh>

using value_types = nvbench::type_list<
  int8_t,
  int16_t,
  int32_t,
  int64_t,
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
  double,
#if _CCCL_HAS_FLOAT128()
  __float128,
#endif
#if _CCCL_HAS_NVFP16() && _CCCL_CTK_AT_LEAST(12, 2)
  cuda::std::complex<__half>,
#endif
#if _CCCL_HAS_NVBF16() && _CCCL_CTK_AT_LEAST(12, 2)
  cuda::std::complex<__nv_bfloat16>,
#endif
  cuda::std::complex<float>,
  cuda::std::complex<double>>;

using op_t = ::cuda::std::plus<>;
#include "warp_reduce_base.cuh"
