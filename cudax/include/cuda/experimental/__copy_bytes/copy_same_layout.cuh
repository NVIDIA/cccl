//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_BYTES_COPY_SAME_LAYOUT
#define _CUDAX__COPY_BYTES_COPY_SAME_LAYOUT

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
#include <cute/numeric/int.hpp>
#include <cute/tensor.hpp>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::detail
{

//! @brief Vectorized same-layout copy kernel using CuTe recast.
//!
//! When source and destination have the same layout, shared memory adds no benefit.
//! Instead, recast both tensors to a wider VecType for vectorized loads/stores and
//! use a grid-stride loop over the recast elements.
//!
//! @tparam VecBytes Vectorization width in bytes (1, 2, 4, 8, or 16)
//! @tparam SrcTensor CuTe tensor type for source (already recast to VecType)
//! @tparam DstTensor CuTe tensor type for destination (already recast to VecType)
template <typename SrcTensor, typename DstTensor>
__global__ void copy_same_layout_kernel(SrcTensor src, DstTensor dst, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < n; i += blockDim.x * gridDim.x)
  {
    dst(i) = src(i);
  }
}

//! @brief Launch the vectorized same-layout copy kernel.
//!
//! Recasts the tensors to a wider VecType based on the computed vectorization
//! width, then launches a grid-stride kernel over the recast elements.
//!
//! @tparam VecBytes Vectorization width in bytes (compile-time template parameter)
//! @tparam T Element type of the original tensors
//! @tparam Layout CuTe layout type (same for src and dst)
template <int VecBytes, typename T, typename Layout>
void launch_copy_same_layout(::cuda::stream_ref stream, T* dst_ptr, const T* src_ptr, const Layout& layout)
{
  using namespace cute;
  using VecType = uint_bit_t<VecBytes * 8>;

  auto src_tensor = recast<VecType>(make_tensor(make_gmem_ptr(src_ptr), layout));
  auto dst_tensor = recast<VecType>(make_tensor(make_gmem_ptr(dst_ptr), layout));

  int n                    = size(src_tensor);
  constexpr int block_size = 256;
  int grid_size            = (n + block_size - 1) / block_size;

  copy_same_layout_kernel<<<grid_size, block_size, 0, stream.get()>>>(src_tensor, dst_tensor, n);
}

} // namespace cuda::experimental::detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_COPY_SAME_LAYOUT
