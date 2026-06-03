//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX___FILL_BYTES_FILL_BYTES_MDSPAN_H
#define __CUDAX___FILL_BYTES_FILL_BYTES_MDSPAN_H

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
#  include <cuda/__mdspan/host_device_mdspan.h>
#  include <cuda/__mdspan/traits.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__mdspan/default_accessor.h>
#  include <cuda/std/__mdspan/mdspan.h>
#  include <cuda/std/__memory/is_sufficiently_aligned.h>
#  include <cuda/std/__type_traits/has_unique_object_representation.h>
#  include <cuda/std/__type_traits/is_const.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_trivially_copyable.h>
#  include <cuda/std/__type_traits/remove_cvref.h>

#  include <cuda/experimental/__copy_bytes/mdspan_to_raw_tensor.cuh>
#  include <cuda/experimental/__copy_bytes/memcpy_batch_tiles.cuh>
#  include <cuda/experimental/__copy_bytes/simplify_paired.cuh>
#  include <cuda/experimental/__copy_bytes/tensor_query.cuh>
#  include <cuda/experimental/__fill_bytes/fill_bytes_mdspan_utils.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _ByteT>
inline constexpr bool __can_fill_bytes_value_v =
  ::cuda::std::is_trivially_copyable_v<_ByteT> && ::cuda::std::has_unique_object_representations_v<_ByteT>
  && (sizeof(_ByteT) == 1 || sizeof(_ByteT) == 2 || sizeof(_ByteT) == 4);

// __half, __nv_bfloat16 don't have a unique object representation
#  if _CCCL_HAS_NVFP16()
template <>
inline constexpr bool __can_fill_bytes_value_v<::__half> = false;
#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()
template <>
inline constexpr bool __can_fill_bytes_value_v<::__nv_bfloat16> = false;
#  endif // _CCCL_HAS_NVBF16()

template <typename _Tp, typename _ByteT>
_CCCL_HOST_API void __fill_bytes_tile(
  _Tp* __ptr, const ::cuda::std::size_t __tile_bytes, const _ByteT __byte_value, const ::cuda::stream_ref __stream)
{
  _CCCL_ASSERT(__tile_bytes % sizeof(_ByteT) == 0,
               "cudax::fill_bytes: destination byte size must be a multiple of the fill value size");
  _CCCL_ASSERT(::cuda::std::is_sufficiently_aligned<sizeof(_ByteT)>(static_cast<void*>(__ptr)),
               "cudax::fill_bytes: destination tile must be sufficiently aligned");
  ::cuda::__driver::__memsetAsync(__ptr, __byte_value, __tile_bytes / sizeof(_ByteT), __stream.get());
}

/***********************************************************************************************************************
 * Public API
 **********************************************************************************************************************/

//! @rst
//! .. _cudax-fill-bytes:
//!
//! Asynchronous mdspan byte fill
//! -----------------------------
//!
//! ``fill_bytes`` asynchronously fills the mapped elements of a device ``mdspan`` with a repeated byte pattern on the
//! given CUDA stream. The pattern is the object representation of a 1-, 2-, or 4-byte fill value. This is a byte
//! operation: it does not assign ``__byte_value`` as an object of the destination element type. For strided layouts,
//! only bytes belonging to mapped destination elements are filled; padding bytes outside the mapping are left
//! unchanged.
//!
//! The operation is enqueued on ``__stream`` and may complete after ``fill_bytes`` returns. Synchronize the stream, or
//! otherwise order dependent work on the same stream, before observing the filled data.
//!
//! - Destination element and fill value types must be trivially copyable.
//! - The fill value type must have unique object representations and size 1, 2, or 4.
//! - The destination element type must not be ``const``.
//! - The destination element size must be a multiple of the fill value size.
//! - The destination element alignment must be at least the fill value size.
//! - Layout policies must be one of the predefined ``cuda::std`` layout policies
//!   (``layout_right``, ``layout_left``, ``layout_stride``) or ``cuda::layout_stride_relaxed``.
//! - Accessor policies must be convertible to ``cuda::std::default_accessor``.
//! - The destination must not have an interleaved stride order.
//! - Zero-size mdspans are no-ops and do not require a non-null data handle.
//!
//! Integer literals use their usual type. For example, ``0`` is an ``int`` and requests a 4-byte pattern fill; use
//! ``cuda::std::uint8_t{0}`` or ``cuda::std::byte{0}`` for a byte pattern fill. The implementation is optimized to
//! maximize the contiguous memory regions to fill.
//!
//! .. code-block:: c++
//!
//!    #include <cuda/experimental/fill_bytes.cuh>
//!
//!    using extents_t = cuda::std::dims<2>;
//!    cuda::device_mdspan<int, extents_t> dst(dst_ptr, extents);
//!    cuda::experimental::fill_bytes(dst, cuda::std::uint32_t{0xFF00FF00}, stream);
//!
//! @endrst
//! @brief Asynchronously fills a device mdspan with a 1-, 2-, or 4-byte pattern.
//!
//! Validates the public preconditions, then dispatches asynchronous memset operations over the mapped destination
//! elements.
//!
//! @param[out] __mdspan Destination device mdspan
//! @param[in] __byte_value Value pattern to fill into the destination
//! @param[in] __stream CUDA stream for the asynchronous fill
//! @throws std::invalid_argument if ``__stream`` is the null stream, or if a non-empty destination has a null data
//! handle, is insufficiently aligned, or has interleaved stride order.
template <typename _Tp, typename _Extents, typename _Layout, typename _Accessor, typename _ByteT>
_CCCL_HOST_API void fill_bytes(::cuda::device_mdspan<_Tp, _Extents, _Layout, _Accessor> __mdspan,
                               const _ByteT __byte_value,
                               const ::cuda::stream_ref __stream)
{
  using __mdspan_t   = ::cuda::std::mdspan<_Tp, _Extents, _Layout, _Accessor>;
  using __value_t    = ::cuda::std::remove_cvref_t<_ByteT>;
  using __accessor_t = ::cuda::std::default_accessor<_Tp>;
  using __extent_t   = typename _Extents::index_type;
  using __stride_t   = __mdspan_stride_t<_Layout, typename __mdspan_t::mapping_type>;

  static_assert(!::cuda::std::is_const_v<_Tp>, "cudax::fill_bytes: element type must not be const");
  static_assert(::cuda::std::is_trivially_copyable_v<_Tp>,
                "cudax::fill_bytes: element type must be trivially copyable");
  static_assert(__can_fill_bytes_value_v<__value_t>,
                "cudax::fill_bytes: fill value type must be trivially copyable with unique object representations and "
                "have size 1, 2, or 4");
  static_assert(sizeof(_Tp) % sizeof(__value_t) == 0,
                "cudax::fill_bytes: element size must be a multiple of the fill value size");
  static_assert(alignof(_Tp) >= sizeof(__value_t),
                "cudax::fill_bytes: element alignment must be at least the fill value size");
  static_assert(::cuda::__is_cuda_mdspan_layout_v<_Layout>,
                "cudax::fill_bytes: LayoutPolicy must be a predefined layout policy");
  static_assert(::cuda::std::is_convertible_v<_Accessor, __accessor_t>,
                "cudax::fill_bytes: AccessorPolicy must be convertible to cuda::std::default_accessor");

  if (__stream.get() == nullptr)
  {
    _CCCL_THROW(::std::invalid_argument, "cudax::fill_bytes: stream must not be nullptr");
  }

  const auto __tensor_size = __mdspan.size();
  if (__tensor_size == 0)
  {
    return;
  }
  if (__mdspan.data_handle() == nullptr)
  {
    _CCCL_THROW(::std::invalid_argument, "cudax::fill_bytes: mdspan data handle must not be nullptr");
  }
  if (!::cuda::std::is_sufficiently_aligned<alignof(_Tp)>(__mdspan.data_handle()))
  {
    _CCCL_THROW(::std::invalid_argument, "cudax::fill_bytes: destination mdspan must be sufficiently aligned");
  }
  if (::cuda::experimental::__has_interleaved_stride_order(__mdspan))
  {
    _CCCL_THROW(::std::invalid_argument,
                "cudax::fill_bytes: destination mdspan must not have interleaved stride order");
  }
  if (__tensor_size == 1) // rank == 0 also falls into this case
  {
    auto* __data_ptr = __mdspan.data_handle();
    if constexpr (::cuda::__is_layout_stride_relaxed_v<_Layout>)
    {
      __data_ptr += __mdspan.mapping().offset();
    }
    ::cuda::experimental::__fill_bytes_tile(__data_ptr, sizeof(_Tp), __byte_value, __stream);
    return;
  }

  constexpr auto __rank = _Extents::rank();
  if constexpr (__rank > 0)
  {
    const auto __raw_tensor = ::cuda::experimental::__to_raw_tensor<__extent_t, __stride_t, __rank>(__mdspan);
    auto __simplified       = ::cuda::experimental::__sort_by_stride(__raw_tensor);
    ::cuda::experimental::__flip_negative_strides_single(__simplified);
    ::cuda::experimental::__coalesce_single(__simplified);

    const bool __stride1      = (__simplified.__strides[0] == 1);
    const auto __tile_size    = __stride1 ? __simplified.__extents[0] : __extent_t{1};
    const auto __final_tensor = (__tile_size > 1) ? __simplified : ::cuda::experimental::__reverse_modes(__simplified);
    const auto __num_tiles    = static_cast<::cuda::std::size_t>(__tensor_size / __tile_size);
    const auto __tile_bytes   = static_cast<::cuda::std::size_t>(__tile_size) * sizeof(_Tp);
    _CCCL_ASSERT(__tensor_size % __tile_size == 0, "cudax::fill_bytes: tensor size must be divisible by tile size");

    __tile_iterator_linearized<__extent_t, __stride_t, _Tp, __rank> __tiles_iterator(__final_tensor, __tile_size);
    for (::cuda::std::size_t __tile_idx = 0; __tile_idx < __num_tiles; ++__tile_idx)
    {
      auto* const __tile_ptr = __tiles_iterator(static_cast<__extent_t>(__tile_idx));
      ::cuda::experimental::__fill_bytes_tile(__tile_ptr, __tile_bytes, __byte_value, __stream);
    }
  }
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX___FILL_BYTES_FILL_BYTES_MDSPAN_H
