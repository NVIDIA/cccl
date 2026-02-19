//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_BYTES_COPY_BYTES_REGISTERS
#define _CUDAX__COPY_BYTES_COPY_BYTES_REGISTERS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/launch>
#include <cuda/std/__type_traits/make_nbit_int.h>

#include <cuda/experimental/__copy_bytes/layout_utils.cuh>

#include <cuda/std/__cccl/prologue.h>
#include <cute/tensor.hpp>

namespace cuda::experimental
{
//! @brief Tiled same-layout copy kernel using cooperative_copy.
//!
//! Each block processes one tile of TileSize vectorized elements along the
//! innermost (stride-1) dimension. Blocks are distributed over
//! (tiles_per_row * outer_size) total tiles, where outer_size is the product
//! of all dimensions except mode 0.
//!
//! Full tiles use cooperative_copy with static shapes for optimal thread
//! partitioning. Boundary tiles (last tile per row) use a bounds-checked loop.
//!
//! Works for arbitrary-rank tensors whose mode 0 has stride Int<1>.
//! For rank-1 tensors, outer_size == 1 and this reduces to a simple 1D tiling.
//!
//! @tparam Config    Kernel launch configuration type (encodes block dimensions)
//! @tparam SrcTensor CuTe tensor type for source (recast to VecType)
//! @tparam DstTensor CuTe tensor type for destination (recast to VecType)
//! @tparam TileSize  Number of VecType elements per tile (compile-time constant)
template <typename Config, typename SrcTensor, typename DstTensor, int TileSize>
__global__ void copy_bytes_registers_tiled_kernel(Config config, SrcTensor src, DstTensor dst, int inner_size)
{
  constexpr int NumThreads = ::cuda::gpu_thread.count(::cuda::block, config);
  uint32_t thread_idx      = ::cuda::gpu_thread.rank(::cuda::block, config);
  int block_idx            = ::cuda::block.rank(::cuda::grid, config);
  int tiles_per_row        = ::cuda::ceil_div(inner_size, TileSize);
  int inner_tile           = block_idx % tiles_per_row;
  int outer_idx            = block_idx / tiles_per_row;
  int inner_offset         = inner_tile * TileSize;
  int flat_offset          = inner_offset + outer_idx * inner_size;
  int remaining            = inner_size - inner_offset;

  if (remaining >= TileSize)
  {
    auto tile_layout = make_static_layout<TileSize>();
    auto src_tile    = make_gmem_tensor(&src(flat_offset), tile_layout);
    auto dst_tile    = make_gmem_tensor(&dst(flat_offset), tile_layout);

    ::cute::cooperative_copy<NumThreads, ::cute::sizeof_bits_v<typename SrcTensor::value_type>>(
      thread_idx, src_tile, dst_tile);
  }
  else
  {
    for (int i = thread_idx; i < remaining; i += NumThreads)
    {
      dst(flat_offset + i) = src(flat_offset + i);
    }
  }
}

//! @brief Launch the vectorized same-layout copy kernel.
//!
//! Recasts the tensors to a wider VecType based on the computed vectorization
//! width. Tiles the innermost (stride-1) dimension with cooperative_copy for
//! all ranks. Blocks are distributed over (tiles_per_row * outer_size).
//!
//! @pre Mode 0 of @p layout must have stride Int<1>.
//!
//! @tparam VecBytes Vectorization width in bytes (compile-time template parameter)
//! @tparam T Element type of the original tensors
//! @tparam Layout CuTe layout type (same for src and dst)
template <int VecBytes, typename T, typename Layout>
void launch_copy_bytes_registers(::cuda::stream_ref stream, const T* src_ptr, T* dst_ptr, const Layout& layout)
{
  using VecType            = ::cuda::std::__make_nbit_uint_t<VecBytes * 8>;
  auto src_tensor          = ::cute::recast<VecType>(make_gmem_tensor(src_ptr, layout));
  auto dst_tensor          = ::cute::recast<VecType>(make_gmem_tensor(dst_ptr, layout));
  constexpr int block_size = 256;
  constexpr int TileSize   = block_size * 4;
  int inner_size           = ::cute::size<0>(src_tensor);
  int outer_size           = ::cute::size(src_tensor) / inner_size;
  int tiles_per_row        = ::cuda::ceil_div(inner_size, TileSize);
  int grid_size            = tiles_per_row * outer_size;
  auto config              = ::cuda::make_config(::cuda::block_dims<block_size>(), ::cuda::grid_dims(grid_size));
  const auto& kernel =
    copy_bytes_registers_tiled_kernel<decltype(config), decltype(src_tensor), decltype(dst_tensor), TileSize>;

  ::cuda::launch(stream, config, kernel, src_tensor, dst_tensor, inner_size);
}

//! @brief Same-layout copy with vectorized register-based path.
//!
//! Preprocesses both layouts independently (sort, coalesce, filter), verifies
//! they match, computes vectorization, reconstructs an optimized layout with
//! Int<1> for the stride-1 mode (required for correct recast), and dispatches
//! to launch_copy_bytes_registers.
//!
//! Falls back to copy_bytes_naive when layouts differ or the configuration
//! is unsupported.
template <typename T, typename SrcLayout, typename DstLayout>
void copy_bytes_registers(
  const T* src, const SrcLayout& src_layout, T* dst, const DstLayout& dst_layout, ::cuda::stream_ref stream)
{
  constexpr int MaxRank  = 8;
  constexpr int SrcR     = decltype(::cute::rank(src_layout))::value;
  constexpr int DstR     = decltype(::cute::rank(dst_layout))::value;
  constexpr int MaxSrcDst = SrcR > DstR ? SrcR : DstR;
  static_assert(MaxSrcDst <= MaxRank, "Layout rank exceeds maximum supported rank");

  ::cuda::std::array<int, MaxRank> src_shapes{};
  ::cuda::std::array<int, MaxRank> src_strides{};
  __extract_layout(src_layout, src_shapes, src_strides);
  sort_modes_by_stride(src_shapes, src_strides, SrcR);
  runtime_coalesce(src_shapes, src_strides, SrcR);

  ::cuda::std::array<int, MaxRank> dst_shapes{};
  ::cuda::std::array<int, MaxRank> dst_strides{};
  __extract_layout(dst_layout, dst_shapes, dst_strides);
  sort_modes_by_stride(dst_shapes, dst_strides, DstR);
  runtime_coalesce(dst_shapes, dst_strides, DstR);

  if (!layouts_match(src_shapes, src_strides, dst_shapes, dst_strides, MaxSrcDst))
  {
    copy_bytes_naive(src, src_layout, dst, dst_layout, stream);
    return;
  }

  ::cuda::std::array<int, MaxRank> eff_s{};
  ::cuda::std::array<int, MaxRank> eff_st{};
  int eff_rank = 0;
  for (int i = 0; i < SrcR; ++i)
  {
    if (src_shapes[i] > 1)
    {
      eff_s[eff_rank]  = src_shapes[i];
      eff_st[eff_rank] = src_strides[i];
      ++eff_rank;
    }
  }

  if (eff_rank == 0)
  {
    auto one = make_static_layout<1>();
    copy_bytes_naive(src, one, dst, one, stream);
    return;
  }

  auto src_vec   = __max_vector_size(src, eff_s, eff_st);
  auto dst_vec   = __max_vector_size(dst, eff_s, eff_st);
  auto vec_bytes = ::cuda::std::min(src_vec, dst_vec);

  auto dispatch = [&](const auto& opt_layout) {
    switch (vec_bytes)
    {
      case 16:
        launch_copy_bytes_registers<16>(stream, src, dst, opt_layout);
        break;
      case 8:
        launch_copy_bytes_registers<8>(stream, src, dst, opt_layout);
        break;
      case 4:
        launch_copy_bytes_registers<4>(stream, src, dst, opt_layout);
        break;
      case 2:
        launch_copy_bytes_registers<2>(stream, src, dst, opt_layout);
        break;
      default:
        launch_copy_bytes_registers<1>(stream, src, dst, opt_layout);
        break;
    }
  };

  if (eff_rank == 1 && eff_st[0] == 1)
  {
    dispatch(::cute::make_layout(::cute::make_shape(eff_s[0]), ::cute::make_stride(::cute::Int<1>{})));
  }
  else if (eff_rank == 2 && eff_st[0] == 1)
  {
    dispatch(
      ::cute::make_layout(::cute::make_shape(eff_s[0], eff_s[1]), ::cute::make_stride(::cute::Int<1>{}, eff_st[1])));
  }
  else if (eff_rank == 3 && eff_st[0] == 1)
  {
    dispatch(::cute::make_layout(
      ::cute::make_shape(eff_s[0], eff_s[1], eff_s[2]), ::cute::make_stride(::cute::Int<1>{}, eff_st[1], eff_st[2])));
  }
  else
  {
    copy_bytes_naive(src, src_layout, dst, dst_layout, stream);
  }
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_COPY_BYTES_REGISTERS
