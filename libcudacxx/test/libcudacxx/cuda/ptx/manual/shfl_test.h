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
  auto res1 = cuda::ptx::shfl_sync_idx(data, pred1, 2 /*idx*/, 0b11111 /*clamp*/, FullMask);
  assert(res1 == 2 && pred1, "shfl_mode_idx failed");

  auto res2 = cuda::ptx::shfl_sync_up(data, pred2, 2 /*offset*/, 0 /*clamp*/, FullMask);
  if (threadIdx.x <= 1)
  {
    assert(res2 == threadIdx.x && !pred2, "shfl_mode_up failed");
  }
  else
  {
    assert(res2 == threadIdx.x - 2 && pred2, "shfl_mode_up failed");
  }

  auto res3 = cuda::ptx::shfl_sync_down(data, pred3, 2 /*offset*/, 0b11111 /*clamp*/, FullMask);
  if (threadIdx.x >= 30)
  {
    assert(res3 == threadIdx.x && !pred3, "shfl_mode_down failed");
  }
  else
  {
    assert(res3 == threadIdx.x + 2 && pred3, "shfl_mode_down failed");
  }

  auto res4 = cuda::ptx::shfl_sync_bfly(data, pred4, 2 /*offset*/, 0b11111 /*clamp*/, FullMask);
  assert(res4 == threadIdx.x ^ 2 && pred4, "shfl_mode_bfly failed");
#endif // __cccl_ptx_isa >= 600
}

__host__ __device__ void test_shfl_full_mask_no_pred()
{
#if __cccl_ptx_isa >= 600 && __CUDA_ARCH__
  constexpr unsigned FullMask = 0xFFFFFFFF;
  auto data                   = threadIdx.x;
  auto res1                   = cuda::ptx::shfl_sync_idx(data, 2 /*idx*/, 0b11111 /*clamp*/, FullMask);
  assert(res1 == 2, "shfl_mode_idx failed");

  auto res2 = cuda::ptx::shfl_sync_up(data, 2 /*offset*/, 0 /*clamp*/, FullMask);
  if (threadIdx.x <= 1)
  {
    assert(res2 == threadIdx.x, "shfl_mode_up failed");
  }
  else
  {
    assert(res2 == threadIdx.x - 2, "shfl_mode_up failed");
  }

  auto res3 = cuda::ptx::shfl_sync_down(data, 2 /*offset*/, 0b11111 /*clamp*/, FullMask);
  if (threadIdx.x >= 30)
  {
    assert(res3 == threadIdx.x, "shfl_mode_down failed");
  }
  else
  {
    assert(res3 == threadIdx.x + 2, "shfl_mode_down failed");
  }

  auto res4 = cuda::ptx::shfl_sync_bfly(data, 2 /*offset*/, 0b11111 /*clamp*/, FullMask);
  assert(res4 == threadIdx.x ^ 2, "shfl_mode_bfly failed");
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
    auto res1 = cuda::ptx::shfl_sync_idx(data, pred1, 2 /*idx*/, 0b11111 /*clamp*/, PartialMask);
    assert(res1 == 2 && pred1, "shfl_mode_idx failed");
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
  auto res1 = cuda::ptx::shfl_sync_idx(data, pred1, 2 /*idx*/, clamp_segmark, FullMask);
  if (threadIdx.x < 16)
  {
    assert(res1 == 2 && pred1, "shfl_mode_idx failed");
  }
  else
  {
    assert(res1 == 16 + 2 && pred1, "shfl_mode_idx failed");
  }

  auto res2 = cuda::ptx::shfl_sync_up(data, pred2, 2 /*offset*/, (max_lane_mask << 8), FullMask);
  if (threadIdx.x <= 1 || threadIdx.x == 16 || threadIdx.x == 17)
  {
    assert(res2 == threadIdx.x && !pred2, "shfl_mode_up failed");
  }
  else
  {
    assert(res2 == threadIdx.x - 2 && pred2, "shfl_mode_up failed");
  }

  auto res3 = cuda::ptx::shfl_sync_down(data, pred3, 2 /*offset*/, clamp_segmark, FullMask);
  if (threadIdx.x == 14 || threadIdx.x == 15 || threadIdx.x >= 30)
  {
    assert(res3 == threadIdx.x && !pred3, "shfl_mode_down failed");
  }
  else
  {
    assert(res3 == threadIdx.x + 2 && pred3, "shfl_mode_down failed");
  }

  auto res4 = cuda::ptx::shfl_sync_bfly(data, pred4, 2 /*offset*/, clamp_segmark, FullMask);
  assert(res4 == threadIdx.x ^ 2 && pred4, "shfl_mode_bfly failed");
#endif // __cccl_ptx_isa >= 600
}
