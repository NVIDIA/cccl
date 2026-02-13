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
#  include <cuda/std/array>

#  include <cuda/experimental/__copy/types.cuh>

#  include <vector>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#  if _CCCL_CTK_AT_LEAST(13, 0)
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

  const int __device_ordinal = ::cuda::__driver::__pointerGetAttribute<::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL>(__src_ptr);
  return ::CUmemcpyAttributes{
    ::CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    ::CUmemLocation{::CU_MEM_LOCATION_TYPE_DEVICE, __device_ordinal},
    ::CUmemLocation{::CU_MEM_LOCATION_TYPE_HOST, 0},
    0};
}

template <typename _SrcTileIterator, typename _DstTileIterator>
_CCCL_HOST_API inline void __memcpy_batch_tiles(
  const _SrcTileIterator& __src_tiles_iterator,
  const _DstTileIterator& __dst_tiles_iterator,
  ::cuda::std::size_t __num_tiles,
  ::cuda::std::size_t __copy_bytes,
  __copy_direction __direction,
  const void* __src_data_handle,
  void* __dst_data_handle,
  ::cuda::stream_ref __stream)
{
  using ::cuda::std::size_t;
  auto __memcpy_batch_async_lambda = [&](auto& __src_ptrs, auto& __dst_ptrs, auto& __sizes) {
    for (size_t __tile_idx = 0; __tile_idx < __num_tiles; ++__tile_idx)
    {
      __src_ptrs[__tile_idx] = __src_tiles_iterator(__tile_idx);
      __dst_ptrs[__tile_idx] = __dst_tiles_iterator(__tile_idx);
      __sizes[__tile_idx]    = __copy_bytes;
    }
    auto __attributes =
      ::cuda::experimental::__get_memcpy_attributes(__direction, __src_data_handle, __dst_data_handle);
    size_t __num_attributes = 0;
    ::cuda::__driver::__memcpyBatchAsync(
      __dst_ptrs.data(),
      __src_ptrs.data(),
      __sizes.data(),
      __num_tiles,
      &__attributes,
      &__num_attributes,
      1,
      __stream.get());
  };

  constexpr size_t __max_tiles = 16;
  if (__num_tiles > __max_tiles)
  {
    ::std::vector<const void*> __src_ptr_vector(__num_tiles);
    ::std::vector<void*> __dst_ptr_vector(__num_tiles);
    ::std::vector<size_t> __sizes(__num_tiles);
    __memcpy_batch_async_lambda(__src_ptr_vector, __dst_ptr_vector, __sizes);
  }
  else
  {
    ::cuda::std::array<const void*, __max_tiles> __src_ptr_vector{};
    ::cuda::std::array<void*, __max_tiles> __dst_ptr_vector{};
    ::cuda::std::array<size_t, __max_tiles> __sizes{};
    __memcpy_batch_async_lambda(__src_ptr_vector, __dst_ptr_vector, __sizes);
  }
}
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_MEMCPY_BATCH_TILES_H
