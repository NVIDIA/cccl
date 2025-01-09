//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: clang && !nvcc

// <cuda/ptx>

__host__ __device__ void test_shfl_full_mask()
{
#if __cccl_ptx_isa >= 600 && __CUDA_ARCH__
  constexpr unsigned FullMask = 0xFFFFFFFF;
  auto data                   = threadIdx.x;
  bool pred1, pred2, pred3, pred4;
  auto res1 = cuda::ptx::shfl_sync(cuda::ptx::shfl_mode_idx, data, 2 /*idx*/, 0b11111 /*clamp*/, FullMask, pred1);
  _CCCL_ASSERT(res1 == 2 && pred1, "shfl_mode_idx failed");

  auto res2 = cuda::ptx::shfl_sync(cuda::ptx::shfl_mode_up, data, 2 /*offset*/, 0 /*clamp*/, FullMask, pred2);
  if (threadIdx.x <= 1)
  {
    _CCCL_ASSERT(res2 == threadIdx.x && !pred2, "shfl_mode_up failed");
  }
  else
  {
    _CCCL_ASSERT(res2 == threadIdx.x - 2 && pred2, "shfl_mode_up failed");
  }

  auto res3 = cuda::ptx::shfl_sync(cuda::ptx::shfl_mode_down, data, 2 /*offset*/, 0b11111 /*clamp*/, FullMask, pred3);
  if (threadIdx.x >= 30)
  {
    _CCCL_ASSERT(res3 == threadIdx.x && !pred3, "shfl_mode_down failed");
  }
  else
  {
    _CCCL_ASSERT(res3 == threadIdx.x + 2 && pred3, "shfl_mode_down failed");
  }

  auto res4 = cuda::ptx::shfl_sync(cuda::ptx::shfl_mode_bfly, data, 2 /*offset*/, 0b11111 /*clamp*/, FullMask, pred4);
  _CCCL_ASSERT(res4 == threadIdx.x ^ 2 && pred4, "shfl_mode_bfly failed");
#endif // __cccl_ptx_isa >= 600
}

__host__ __device__ void test_shfl_partial_mask()
{
#if __cccl_ptx_isa >= 600 && __CUDA_ARCH__
  constexpr unsigned PartialMask = 0b1111;
  auto data                      = threadIdx.x;
  bool pred1;
  if (threadIdx.x <= 3)
  {
    auto res1 = cuda::ptx::shfl_sync(cuda::ptx::shfl_mode_idx, data, 2 /*idx*/, 0b11111 /*clamp*/, PartialMask, pred1);
    _CCCL_ASSERT(res1 == 2 && pred1, "shfl_mode_idx failed");
  }
#endif // __cccl_ptx_isa >= 600
}

__host__ __device__ void test_shfl_partial_warp()
{
#if __cccl_ptx_isa >= 600 && __CUDA_ARCH__
  constexpr unsigned FullMask = 0xFFFFFFFF;
  unsigned max_lane_mask      = 16;
  unsigned clamp              = 0b11111;
  unsigned clamp_segmark      = (max_lane_mask << 8) | clamp;
  auto data                   = threadIdx.x;
  bool pred1, pred2, pred3, pred4;
  auto res1 = cuda::ptx::shfl_sync(cuda::ptx::shfl_mode_idx, data, 2 /*idx*/, clamp_segmark, FullMask, pred1);
  if (threadIdx.x < 16)
  {
    _CCCL_ASSERT(res1 == 2 && pred1, "shfl_mode_idx failed");
  }
  else
  {
    _CCCL_ASSERT(res1 == 16 + 2 && pred1, "shfl_mode_idx failed");
  }

  auto res2 = cuda::ptx::shfl_sync(cuda::ptx::shfl_mode_up, data, 2 /*offset*/, (max_lane_mask << 8), FullMask, pred2);
  printf("%d:  res2 = %d, pred2 = %d\n", threadIdx.x, res2, pred2);
  if (threadIdx.x <= 1 || threadIdx.x == 16 || threadIdx.x == 17)
  {
    _CCCL_ASSERT(res2 == threadIdx.x && !pred2, "shfl_mode_up failed");
  }
  else
  {
    _CCCL_ASSERT(res2 == threadIdx.x - 2 && pred2, "shfl_mode_up failed");
  }

  auto res3 = cuda::ptx::shfl_sync(cuda::ptx::shfl_mode_down, data, 2 /*offset*/, clamp_segmark, FullMask, pred3);
  if (threadIdx.x == 14 || threadIdx.x == 15 || threadIdx.x >= 30)
  {
    _CCCL_ASSERT(res3 == threadIdx.x && !pred3, "shfl_mode_down failed");
  }
  else
  {
    _CCCL_ASSERT(res3 == threadIdx.x + 2 && pred3, "shfl_mode_down failed");
  }

  auto res4 = cuda::ptx::shfl_sync(cuda::ptx::shfl_mode_bfly, data, 2 /*offset*/, clamp_segmark, FullMask, pred4);
  _CCCL_ASSERT(res4 == threadIdx.x ^ 2 && pred4, "shfl_mode_bfly failed");
#endif // __cccl_ptx_isa >= 600
}
