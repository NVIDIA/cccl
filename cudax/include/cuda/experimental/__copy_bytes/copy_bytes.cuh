//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_BYTES_COPY_BYTES
#define _CUDAX__COPY_BYTES_COPY_BYTES

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__exception/cuda_error.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include <cuda/experimental/__copy_bytes/copy_diff_layout.cuh>
#include <cuda/experimental/__copy_bytes/copy_same_layout.cuh>
#include <cuda/experimental/__copy_bytes/layout_utils.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

//! @brief Device kernel that copies elements between two tensors with potentially different layouts.
//!
//! Each thread processes elements in a grid-stride loop. CuTe handles the mapping from
//! a linear index to multi-dimensional coordinates according to each tensor's layout.
template <typename SrcTensor, typename DstTensor>
__global__ void copy_bytes_kernel(SrcTensor src, DstTensor dst, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < n; i += blockDim.x * gridDim.x)
  {
    dst(i) = src(i);
  }
}

namespace detail
{

//! @brief Launch the naive fallback kernel.
template <typename T, typename SrcLayout, typename DstLayout>
void launch_naive_copy(
  ::cuda::stream_ref stream, T* dst, const T* src, const SrcLayout& src_layout, const DstLayout& dst_layout)
{
  using namespace cute;
  auto src_tensor          = make_tensor(make_gmem_ptr(src), src_layout);
  auto dst_tensor          = make_tensor(make_gmem_ptr(dst), dst_layout);
  int n                    = size(src_layout);
  constexpr int block_size = 256;
  int grid_size            = (n + block_size - 1) / block_size;

  copy_bytes_kernel<<<grid_size, block_size, 0, stream.get()>>>(src_tensor, dst_tensor, n);
}

//! @brief Dispatch same-layout copy with the appropriate vectorization width.
//!
//! Constructs an optimized layout with Int<1> for the stride-1 mode
//! (required for correct recast behavior) and dispatches based on vec_bytes.
template <typename T, typename Layout>
void dispatch_same_layout(::cuda::stream_ref stream, T* dst, const T* src, const Layout& layout, int vec_bytes)
{
  switch (vec_bytes)
  {
    case 16:
      launch_copy_same_layout<16>(stream, dst, src, layout);
      break;
    case 8:
      launch_copy_same_layout<8>(stream, dst, src, layout);
      break;
    case 4:
      launch_copy_same_layout<4>(stream, dst, src, layout);
      break;
    case 2:
      launch_copy_same_layout<2>(stream, dst, src, layout);
      break;
    default:
      launch_copy_same_layout<1>(stream, dst, src, layout);
      break;
  }
}

} // namespace detail

//! @brief Copy elements between two device-memory tensors described by CuTe layouts.
//!
//! Copies elements from @p src to @p dst, where both pointers refer to device memory.
//! The source and destination may have different strides (memory layouts), but must
//! have the same logical shape (i.e., the same number of elements).
//!
//! The implementation selects an optimized kernel path based on layout analysis:
//! - **Same layout, contiguous**: vectorized copy using recast to wider types
//! - **Different layout, rank-2**: shared-memory tiled copy with swizzle
//! - **Fallback**: naive grid-stride loop for all other cases
//!
//! @par Example
//! @code
//! #include <cuda/experimental/copy_bytes.cuh>
//! #include <cute/layout.hpp>
//!
//! using namespace cute;
//!
//! // 2D tensor: 128 rows x 256 cols
//! auto shape      = make_shape(128, 256);
//! auto src_layout = make_layout(shape, make_stride(256, 1));  // row-major
//! auto dst_layout = make_layout(shape, make_stride(1, 128));  // col-major
//!
//! cuda::experimental::copy_bytes(stream, dst_ptr, dst_layout, src_ptr, src_layout);
//! @endcode
//!
//! @tparam T          Element type of the tensors
//! @tparam SrcLayout  CuTe layout type for the source tensor
//! @tparam DstLayout  CuTe layout type for the destination tensor
//!
//! @param stream      CUDA stream on which to launch the copy kernel
//! @param dst         Pointer to the destination tensor in device memory
//! @param dst_layout  CuTe layout describing the shape and strides of @p dst
//! @param src         Pointer to the source tensor in device memory
//! @param src_layout  CuTe layout describing the shape and strides of @p src
template <typename T, typename SrcLayout, typename DstLayout>
void copy_bytes(
  ::cuda::stream_ref stream, T* dst, DstLayout dst_layout, const T* src, SrcLayout src_layout)
{
  using namespace cute;

  _CCCL_ASSERT(
    size(src_layout) == size(dst_layout), "Source and destination layouts must have the same number of elements");

  constexpr int MaxRank = 8;
  constexpr int R       = decltype(rank(src_layout))::value;
  static_assert(R <= MaxRank, "Layout rank exceeds maximum supported rank");

  // Step 1: Extract original shapes/strides and compare BEFORE preprocessing.
  // Sorting and coalescing independently can make different layouts appear identical.
  int src_s[MaxRank], src_st[MaxRank], dst_s[MaxRank], dst_st[MaxRank];
  detail::extract_layout(src_layout, src_s, src_st);
  detail::extract_layout(dst_layout, dst_s, dst_st);

  bool same_layout = detail::layouts_match(src_s, src_st, dst_s, dst_st, R);

  if (same_layout)
  {
    // Same-layout path: sort + coalesce for vectorization optimization
    detail::sort_modes_by_stride(src_s, src_st, R);
    detail::runtime_coalesce(src_s, src_st, R);

    // Collect effective (non-trivial) modes
    int eff_s[MaxRank], eff_st[MaxRank];
    int eff_rank = 0;
    for (int i = 0; i < R; ++i)
    {
      if (src_s[i] > 1)
      {
        eff_s[eff_rank]  = src_s[i];
        eff_st[eff_rank] = src_st[i];
        ++eff_rank;
      }
    }

    if (eff_rank == 0)
    {
      auto one_layout = make_layout(make_shape(Int<1>{}), make_stride(Int<1>{}));
      detail::launch_naive_copy(stream, dst, src, one_layout, one_layout);
    }
    else
    {
      int vec_bytes = detail::compute_vec_bytes(src, dst, eff_s, eff_st, eff_rank, sizeof(T));

      // Reconstruct layout with Int<1> for the stride-1 mode (required for correct recast).
      // After sorting, mode 0 has the smallest stride.
      if (eff_rank == 1 && eff_st[0] == 1)
      {
        auto opt = make_layout(make_shape(eff_s[0]), make_stride(Int<1>{}));
        detail::dispatch_same_layout(stream, dst, src, opt, vec_bytes);
      }
      else if (eff_rank == 2 && eff_st[0] == 1)
      {
        auto opt = make_layout(make_shape(eff_s[0], eff_s[1]), make_stride(Int<1>{}, eff_st[1]));
        detail::dispatch_same_layout(stream, dst, src, opt, vec_bytes);
      }
      else if (eff_rank == 3 && eff_st[0] == 1)
      {
        auto opt = make_layout(
          make_shape(eff_s[0], eff_s[1], eff_s[2]), make_stride(Int<1>{}, eff_st[1], eff_st[2]));
        detail::dispatch_same_layout(stream, dst, src, opt, vec_bytes);
      }
      else
      {
        detail::launch_naive_copy(stream, dst, src, src_layout, dst_layout);
      }
    }
  }
  else if (R == 2)
  {
    // Different-layout path: shared-memory tiled copy (rank-2)
    // Use the original shapes/strides (not coalesced) to preserve 2D structure
    int M        = src_s[0];
    int N        = src_s[1];
    auto src_opt = make_layout(make_shape(M, N), make_stride(src_st[0], src_st[1]));
    auto dst_opt = make_layout(make_shape(M, N), make_stride(dst_st[0], dst_st[1]));
    detail::launch_copy_diff_layout(stream, dst, src, src_opt, dst_opt, M, N);
  }
  else
  {
    // Unsupported configuration: fall back to naive kernel
    detail::launch_naive_copy(stream, dst, src, src_layout, dst_layout);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    ::cuda::__throw_cuda_error(err, "copy_bytes kernel launch failed");
  }
}

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_COPY_BYTES
