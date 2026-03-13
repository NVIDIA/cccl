//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_MEMCPY_BATCH_TILES_H
#define __CUDAX_COPY_MEMCPY_BATCH_TILES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__driver/driver_api.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__numeric/exclusive_scan.h>
#  include <cuda/std/array>

#  include <cuda/experimental/__copy_bytes/types.cuh>

#  include <vector>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Iterator that maps a linear tile index to a pointer into a strided raw tensor.
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
struct __tile_iterator_linearized
{
  const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank> __tensor_;
  ::cuda::std::array<_ExtentT, _MaxRank> __extent_products_;
  const _ExtentT __contiguous_size_;

  //! @brief Constructs the iterator from a raw tensor and contiguous tile size.
  //!
  //! @param[in] __tensor          Raw tensor descriptor
  //! @param[in] __contiguous_size Number of contiguous elements per tile
  _CCCL_HOST_API explicit __tile_iterator_linearized(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor,
                                                     _ExtentT __contiguous_size) noexcept
      : __tensor_{__tensor}
      , __extent_products_{}
      , __contiguous_size_{__contiguous_size}
  {
    // Precomputes exclusive prefix products of extents so that each `operator()` call decomposes a flat index into
    // multi-dimensional coordinates and computes the corresponding byte offset.
    ::cuda::std::exclusive_scan(
      __tensor.__extents.data(),
      __tensor.__extents.data() + __tensor.__rank,
      __extent_products_.data(),
      _ExtentT{1},
      ::cuda::std::multiplies<>{});
  }

  //! @brief Returns a pointer to the first element of the tile at @p __tile_idx.
  //!
  //! @param[in] __tile_idx linear tile index
  //! @return Pointer into the tensor at the computed multi-dimensional offset
  [[nodiscard]] _CCCL_HOST_API _Tp* operator()(_ExtentT __tile_idx) const noexcept
  {
    using __uextent_t     = ::cuda::std::make_unsigned_t<_ExtentT>;
    const auto __index    = __tile_idx * __contiguous_size_;
    const auto& __extents = __tensor_.__extents;
    const auto& __strides = __tensor_.__strides;
    if (__tensor_.__rank == 1)
    {
      return __tensor_.__data + __index * __strides[0];
    }
    const auto __extent0 = static_cast<__uextent_t>(__extents[0]);
    _StrideT __offset    = (__index % __extent0) * __strides[0]; // __extent_products_[0] == 1
    for (::cuda::std::size_t __i = 1; __i < __tensor_.__rank; ++__i)
    {
      const auto __extent_product = static_cast<__uextent_t>(__extent_products_[__i]);
      const auto __coord          = static_cast<_StrideT>((__index / __extent_product) % __extents[__i]);
      __offset += __coord * __strides[__i];
    }
    return __tensor_.__data + __offset;
  }
};

#  if _CCCL_CTK_AT_LEAST(13, 0)

//! @brief Builds the `CUmemcpyAttributes` descriptor for a batch async memcpy.
//!
//! @param[in] __direction Copy direction (host-to-device or device-to-host)
//! @param[in] __src_ptr   Source pointer (used to query device ordinal for D2H)
//! @param[in] __dst_ptr   Destination pointer (used to query device ordinal for H2D)
//! @return Populated `CUmemcpyAttributes` struct
[[nodiscard]] _CCCL_HOST_API inline ::CUmemcpyAttributes
__get_memcpy_attributes(__copy_direction __direction, const void* __src_ptr, const void* __dst_ptr) noexcept
{
  if (__direction == __copy_direction::__host_to_device)
  {
    const int __device_ordinal =
      ::cuda::__driver::__pointerGetAttribute<::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL>(__dst_ptr);
    return ::CUmemcpyAttributes{
      ::CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
      ::CUmemLocation{::CU_MEM_LOCATION_TYPE_HOST, 0},
      ::CUmemLocation{::CU_MEM_LOCATION_TYPE_DEVICE, __device_ordinal},
      0};
  }
  const int __device_ordinal =
    ::cuda::__driver::__pointerGetAttribute<::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL>(__src_ptr);
  return ::CUmemcpyAttributes{
    ::CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    ::CUmemLocation{::CU_MEM_LOCATION_TYPE_DEVICE, __device_ordinal},
    ::CUmemLocation{::CU_MEM_LOCATION_TYPE_HOST, 0},
    0};
}

#  endif // _CCCL_CTK_AT_LEAST(13, 0)

//! @brief Submits an asynchronous batch memcpy for every tile.
//!
//! - Uses `cuMemcpyBatchAsync` on CTK 13.0+ with stack-allocated arrays for small tile counts,
//!   falling back to heap allocation when the count exceeds a fixed threshold.
//! - On older toolkits, issues individual `cuMemcpyAsync` calls per tile.
//!
//! @param[in] __src_tiles_iterator Tile iterator for the source tensor
//! @param[in] __dst_tiles_iterator Tile iterator for the destination tensor
//! @param[in] __num_tiles          Number of tiles to copy
//! @param[in] __copy_size_bytes    Byte size of each tile
//! @param[in] __direction          Copy direction
//! @param[in] __src_data_handle    Source base pointer (for attribute query)
//! @param[in] __dst_data_handle    Destination base pointer (for attribute query)
//! @param[in] __stream             CUDA stream
template <typename _SrcTileIterator, typename _DstTileIterator>
_CCCL_HOST_API inline void __memcpy_batch_tiles(
  const _SrcTileIterator& __src_tiles_iterator,
  const _DstTileIterator& __dst_tiles_iterator,
  ::cuda::std::size_t __num_tiles,
  ::cuda::std::size_t __copy_size_bytes,
  [[maybe_unused]] __copy_direction __direction,
  [[maybe_unused]] const void* __src_data_handle,
  [[maybe_unused]] void* __dst_data_handle,
  ::cuda::stream_ref __stream)
{
  using ::cuda::std::size_t;
#  if _CCCL_CTK_AT_LEAST(13, 0)
  auto __attributes = ::cuda::experimental::__get_memcpy_attributes(__direction, __src_data_handle, __dst_data_handle);
  const auto __memcpy_batch_async_lambda = [&](auto __src_ptrs, auto __dst_ptrs, auto __sizes) {
    for (size_t __tile_idx = 0; __tile_idx < __num_tiles; ++__tile_idx)
    {
      __src_ptrs[__tile_idx] = __src_tiles_iterator(__tile_idx);
      __dst_ptrs[__tile_idx] = __dst_tiles_iterator(__tile_idx);
      __sizes[__tile_idx]    = __copy_size_bytes;
    }
    size_t __attribute_indices = 0;
    ::cuda::__driver::__memcpyBatchAsync(
      __dst_ptrs,
      __src_ptrs,
      __sizes,
      __num_tiles,
      &__attributes,
      &__attribute_indices,
      /*num_attributes=*/1,
      __stream.get());
  };
  constexpr size_t __max_tiles = 16;
  if (__num_tiles > __max_tiles)
  {
    auto __src_ptrs = new const void*[__num_tiles];
    auto __dst_ptrs = new void*[__num_tiles];
    auto __sizes    = new size_t[__num_tiles];
    __memcpy_batch_async_lambda(__src_ptrs, __dst_ptrs, __sizes);
    delete[] __src_ptrs;
    delete[] __dst_ptrs;
    delete[] __sizes;
  }
  else
  {
    ::cuda::std::array<const void*, __max_tiles> __src_ptr_array{};
    ::cuda::std::array<void*, __max_tiles> __dst_ptr_array{};
    ::cuda::std::array<size_t, __max_tiles> __sizes{};
    __memcpy_batch_async_lambda(__src_ptr_array.data(), __dst_ptr_array.data(), __sizes.data());
  }
#  else
  for (size_t __tile_idx = 0; __tile_idx < __num_tiles; ++__tile_idx)
  {
    const auto __src_ptr = static_cast<const void*>(__src_tiles_iterator(__tile_idx));
    const auto __dst_ptr = static_cast<void*>(__dst_tiles_iterator(__tile_idx));
    ::cuda::__driver::__memcpyAsync(__dst_ptr, __src_ptr, __copy_size_bytes, __stream.get());
  }
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_MEMCPY_BATCH_TILES_H
