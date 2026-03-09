//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_MDSPAN_D2H_H2D_H
#define __CUDAX_COPY_MDSPAN_D2H_H2D_H

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
#  include <cuda/std/__algorithm/stable_sort.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__memory/is_sufficiently_aligned.h>
#  include <cuda/std/__numeric/gcd_lcm.h>
#  include <cuda/std/__type_traits/common_type.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/make_signed.h>
#  include <cuda/std/array>

#  include <cuda/experimental/__copy/max_common_layout.cuh>
#  include <cuda/experimental/__copy/memcpy_batch_tiles.cuh>
#  include <cuda/experimental/__copy/utils.cuh>

#  include <stdexcept>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _Ep, typename _Sp, typename _Tp, ::cuda::std::size_t _MaxRank>
struct __tile_iterator_linearized
{
  const __raw_tensor<_Ep, _Sp, _Tp, _MaxRank> __tensor_original_;
  const __raw_tensor<_Ep, _Sp, _Tp, _MaxRank> __tensor_pair_sorted_;
  const ::cuda::std::size_t __contiguous_size_;
  const bool __use_pair_order_;

  _CCCL_HOST_API explicit __tile_iterator_linearized(
    const __raw_tensor<_Ep, _Sp, _Tp, _MaxRank>& __tensor_original,
    const __raw_tensor<_Ep, _Sp, _Tp, _MaxRank>& __tensor_pair_sorted,
    ::cuda::std::size_t __contiguous_size,
    bool __use_pair_order) noexcept
      : __tensor_original_{__tensor_original}
      , __tensor_pair_sorted_{__tensor_pair_sorted}
      , __contiguous_size_{__contiguous_size}
      , __use_pair_order_{__use_pair_order}
  {}

  [[nodiscard]] _CCCL_HOST_API _Tp* operator()(::cuda::std::size_t __tile_idx) const noexcept
  {
    const auto& __tensor  = __use_pair_order_ ? __tensor_pair_sorted_ : __tensor_original_;
    const auto& __extents = __tensor.__extents;
    const auto& __strides = __tensor.__strides;
    const auto __rank     = __tensor.__rank;
    auto __index          = __tile_idx * __contiguous_size_;
    _Sp __offset          = 0;
    if (__use_pair_order_)
    {
      for (::cuda::std::size_t __i = 0; __i < __rank; ++__i)
      {
        __offset += static_cast<_Sp>(__index % __extents[__i]) * __strides[__i];
        __index /= __extents[__i];
      }
    }
    else
    {
      for (::cuda::std::size_t __i = __rank; __i > 0; --__i)
      {
        const auto __idx = __i - 1;
        __offset += static_cast<_Sp>(__index % __extents[__idx]) * __strides[__idx];
        __index /= __extents[__idx];
      }
    }
    return __tensor.__data + __offset;
  }
};

template <typename _Ep, typename _Sp>
struct __paired_mode
{
  using __unsigned_extent_t = ::cuda::std::make_unsigned_t<_Ep>;

  __unsigned_extent_t __src_extent;
  __unsigned_extent_t __dst_extent;
  _Sp __src_stride;
  _Sp __dst_stride;
};

template <typename _Ep, typename _Sp>
struct __compare_mode_by_dst_stride
{
  [[nodiscard]] _CCCL_HOST_API constexpr bool
  operator()(const __paired_mode<_Ep, _Sp>& __lhs, const __paired_mode<_Ep, _Sp>& __rhs) const noexcept
  {
    return ::cuda::experimental::__abs_integer(__lhs.__dst_stride)
         < ::cuda::experimental::__abs_integer(__rhs.__dst_stride);
  }
};

template <typename _Ep, typename _Sp, typename _TpSrc, typename _TpDst, ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void __sort_by_stride_paired_generalized(
  __raw_tensor<_Ep, _Sp, _TpSrc, _MaxRank>& __src, __raw_tensor<_Ep, _Sp, _TpDst, _MaxRank>& __dst) noexcept
{
  using ::cuda::std::size_t;
  using __mode_t = __paired_mode<_Ep, _Sp>;
  _CCCL_ASSERT(__src.__rank == __dst.__rank, "Source and destination ranks must be equal");
  ::cuda::std::array<__mode_t, _MaxRank> __modes{};
  const auto __rank = __src.__rank;
  for (size_t __i = 0; __i < __rank; ++__i)
  {
    __modes[__i] = {__src.__extents[__i], __dst.__extents[__i], __src.__strides[__i], __dst.__strides[__i]};
  }
  ::cuda::std::stable_sort(__modes.begin(), __modes.begin() + __rank, __compare_mode_by_dst_stride<_Ep, _Sp>{});
  for (size_t __i = 0; __i < __rank; ++__i)
  {
    __src.__extents[__i] = __modes[__i].__src_extent;
    __dst.__extents[__i] = __modes[__i].__dst_extent;
    __src.__strides[__i] = __modes[__i].__src_stride;
    __dst.__strides[__i] = __modes[__i].__dst_stride;
  }
}

template <typename _Ep, typename _Sp, typename _TpSrc, typename _TpDst, ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void __flip_negative_strides_paired_generalized(
  __raw_tensor<_Ep, _Sp, _TpSrc, _MaxRank>& __src, __raw_tensor<_Ep, _Sp, _TpDst, _MaxRank>& __dst) noexcept
{
  _CCCL_ASSERT(__src.__rank == __dst.__rank, "Source and destination ranks must be equal");
  for (::cuda::std::size_t __i = 0; __i < __src.__rank; ++__i)
  {
    if (__src.__strides[__i] < 0 && __dst.__strides[__i] < 0)
    {
      const auto __src_adjustment = static_cast<_Sp>(__src.__extents[__i] - 1) * __src.__strides[__i];
      const auto __dst_adjustment = static_cast<_Sp>(__dst.__extents[__i] - 1) * __dst.__strides[__i];
      __src.__data += __src_adjustment;
      __dst.__data += __dst_adjustment;
      __src.__strides[__i] = -__src.__strides[__i];
      __dst.__strides[__i] = -__dst.__strides[__i];
    }
  }
}

template <typename _Ep, typename _Sp, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr __raw_tensor<_Ep, _Sp, _Tp, _MaxRank>
__erase_order(const __raw_tensor_ordered<_Ep, _Sp, _Tp, _MaxRank>& __tensor) noexcept
{
  __raw_tensor<_Ep, _Sp, _Tp, _MaxRank> __result{__tensor.__data, __tensor.__rank, {}, {}};
  for (::cuda::std::size_t __i = 0; __i < __tensor.__rank; ++__i)
  {
    __result.__extents[__i] = __tensor.__extents[__i];
    __result.__strides[__i] = __tensor.__strides[__i];
  }
  return __result;
}

template <typename _Ep, typename _Sp, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr bool
__is_reverse_order(const __raw_tensor_ordered<_Ep, _Sp, _Tp, _MaxRank>& __tensor) noexcept
{
  for (::cuda::std::size_t __i = 0; __i < __tensor.__rank; ++__i)
  {
    if (__tensor.__orders[__i] != (__tensor.__rank - 1 - __i))
    {
      return false;
    }
  }
  return true;
}

template <typename _Ep, typename _Sp, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr bool
__is_contiguous_in_original_order(const __raw_tensor<_Ep, _Sp, _Tp, _MaxRank>& __tensor) noexcept
{
  _Sp __expected_stride = 1;
  for (::cuda::std::size_t __i = __tensor.__rank; __i > 0; --__i)
  {
    const auto __idx = __i - 1;
    if (__tensor.__strides[__idx] != __expected_stride)
    {
      return false;
    }
    __expected_stride *= static_cast<_Sp>(__tensor.__extents[__idx]);
  }
  return true;
}

template <typename _Ep, typename _Sp, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::size_t
__tensor_size(const __raw_tensor<_Ep, _Sp, _Tp, _MaxRank>& __tensor) noexcept
{
  ::cuda::std::size_t __size = 1;
  for (::cuda::std::size_t __i = 0; __i < __tensor.__rank; ++__i)
  {
    __size *= static_cast<::cuda::std::size_t>(__tensor.__extents[__i]);
  }
  return __size;
}

template <typename _Ep, typename _Sp, typename _TpLow, typename _TpHigh, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr bool __try_factorize_to_match_sorted(
  const __raw_tensor<_Ep, _Sp, _TpLow, _MaxRank>& __low,
  const __raw_tensor_ordered<_Ep, _Sp, _TpHigh, _MaxRank>& __high_sorted,
  __raw_tensor<_Ep, _Sp, _TpLow, _MaxRank>& __low_factorized) noexcept
{
  if (!::cuda::experimental::__is_contiguous_in_original_order(__low))
  {
    return false;
  }
  if (!::cuda::experimental::__is_reverse_order(__high_sorted))
  {
    return false;
  }

  auto __remaining_size = ::cuda::experimental::__tensor_size(__low);
  _Sp __current_stride  = 1;
  __low_factorized      = {__low.__data, __high_sorted.__rank, {}, {}};
  for (::cuda::std::size_t __i = 0; __i < __high_sorted.__rank; ++__i)
  {
    const auto __target_extent = static_cast<::cuda::std::size_t>(__high_sorted.__extents[__i]);
    if ((__target_extent == 0) || ((__remaining_size % __target_extent) != 0))
    {
      return false;
    }
    __low_factorized.__extents[__i] = __high_sorted.__extents[__i];
    __low_factorized.__strides[__i] = __current_stride;
    __current_stride *= static_cast<_Sp>(__target_extent);
    __remaining_size /= __target_extent;
  }
  return __remaining_size == 1;
}

template <typename _Ep, typename _Sp>
[[nodiscard]] _CCCL_HOST_API constexpr bool __can_merge_paired_mode(
  _Ep __prev_extent, _Sp __prev_stride, _Ep __curr_extent, _Sp __curr_stride, _Sp& __merged_stride) noexcept
{
  if (__prev_extent == 1)
  {
    __merged_stride = __curr_stride;
    return true;
  }
  if (__curr_extent == 1)
  {
    __merged_stride = __prev_stride;
    return true;
  }

  const auto __prev_extent_stride = static_cast<_Sp>(__prev_extent) * __prev_stride;
  if (__prev_extent_stride == __curr_stride)
  {
    __merged_stride = __prev_stride;
    return true;
  }
  return false;
}

template <typename _Ep, typename _Sp, typename _TpSrc, typename _TpDst, ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void __coalesce_paired_generalized(__raw_tensor<_Ep, _Sp, _TpSrc, _MaxRank>& __src,
                                                  __raw_tensor<_Ep, _Sp, _TpDst, _MaxRank>& __dst) noexcept
{
  _CCCL_ASSERT(__src.__rank == __dst.__rank, "Source and destination ranks must be equal");
  if (__src.__rank <= 1)
  {
    return;
  }

  ::cuda::std::size_t __out = 1;
  for (::cuda::std::size_t __i = 1; __i < __src.__rank; ++__i)
  {
    _Sp __src_merged_stride{};
    _Sp __dst_merged_stride{};
    const bool __src_contiguous = ::cuda::experimental::__can_merge_paired_mode(
      __src.__extents[__out - 1],
      __src.__strides[__out - 1],
      __src.__extents[__i],
      __src.__strides[__i],
      __src_merged_stride);
    const bool __dst_contiguous = ::cuda::experimental::__can_merge_paired_mode(
      __dst.__extents[__out - 1],
      __dst.__strides[__out - 1],
      __dst.__extents[__i],
      __dst.__strides[__i],
      __dst_merged_stride);
    if (__src_contiguous && __dst_contiguous)
    {
      __src.__extents[__out - 1] *= __src.__extents[__i];
      __dst.__extents[__out - 1] *= __dst.__extents[__i];
      __src.__strides[__out - 1] = __src_merged_stride;
      __dst.__strides[__out - 1] = __dst_merged_stride;
      continue;
    }

    __src.__extents[__out] = __src.__extents[__i];
    __dst.__extents[__out] = __dst.__extents[__i];
    __src.__strides[__out] = __src.__strides[__i];
    __dst.__strides[__out] = __dst.__strides[__i];
    ++__out;
  }

  __src.__rank = __out;
  __dst.__rank = __out;
}

template <typename _TpIn,
          typename _ExtentsIn,
          typename _LayoutPolicyIn,
          typename _AccessorPolicyIn,
          typename _TpOut,
          typename _ExtentsOut,
          typename _LayoutPolicyOut,
          typename _AccessorPolicyOut>
_CCCL_HOST_API void __copy_bytes_impl(
  ::cuda::std::mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn> __src,
  ::cuda::std::mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut> __dst,
  [[maybe_unused]] __copy_direction __direction,
  ::cuda::stream_ref __stream)
{
  namespace cudax = ::cuda::experimental;
  static_assert(::cuda::std::is_same_v<::cuda::std::remove_cv_t<_TpIn>, ::cuda::std::remove_cv_t<_TpOut>>,
                "TpIn and TpOut must be the same type");
  static_assert(::cuda::std::is_trivially_copyable_v<_TpIn>, "TpIn must be trivially copyable");
  static_assert(!::cuda::std::is_const_v<_TpOut>, "TpOut must not be const");
  static_assert(::cuda::__is_cuda_mdspan_layout_v<_LayoutPolicyIn>,
                "LayoutPolicyIn must be a predefined layout policy");
  static_assert(::cuda::__is_cuda_mdspan_layout_v<_LayoutPolicyOut>,
                "LayoutPolicyOut must be a predefined layout policy");
  using __default_accessor_in  = ::cuda::std::default_accessor<_TpIn>;
  using __default_accessor_out = ::cuda::std::default_accessor<_TpOut>;
  static_assert(::cuda::std::is_convertible_v<_AccessorPolicyIn, __default_accessor_in>,
                "AccessorPolicyIn must be convertible to cuda::std::default_accessor");
  static_assert(::cuda::std::is_convertible_v<_AccessorPolicyOut, __default_accessor_out>,
                "AccessorPolicyOut must be convertible to cuda::std::default_accessor");
  if (__src.size() != __dst.size())
  {
    _CCCL_THROW(::std::invalid_argument, "mdspans must have the same size");
  }
  const auto __tensor_size = __src.size();
  if (__tensor_size == 0)
  {
    return;
  }
  if (__src.data_handle() == nullptr || __dst.data_handle() == nullptr)
  {
    _CCCL_THROW(::std::invalid_argument, "mdspan data handle must not be nullptr");
  }
  if (!::cuda::std::is_sufficiently_aligned<alignof(_TpIn)>(__src.data_handle()))
  {
    _CCCL_THROW(::std::invalid_argument, "source mdspan must be sufficiently aligned");
  }
  if (!::cuda::std::is_sufficiently_aligned<alignof(_TpOut)>(__dst.data_handle()))
  {
    _CCCL_THROW(::std::invalid_argument, "destination mdspan must be sufficiently aligned");
  }
  if (cudax::__has_interleaved_stride_order(__dst))
  {
    _CCCL_THROW(::std::invalid_argument, "destination mdspan must not have interleaved stride order");
  }
  if (__tensor_size == 1)
  {
    ::cuda::__driver::__memcpyAsync(__dst.data_handle(), __src.data_handle(), sizeof(_TpIn), __stream.get());
    return;
  }
  if constexpr (_ExtentsIn::rank() > 0 && _ExtentsOut::rank() > 0)
  {
    using ::cuda::std::size_t;
    using __extent_t = ::cuda::std::common_type_t<typename _ExtentsIn::index_type, typename _ExtentsOut::index_type>;
    using __stride_t = ::cuda::std::common_type_t<cudax::__mdspan_stride_t<_ExtentsIn, _LayoutPolicyIn>,
                                                  cudax::__mdspan_stride_t<_ExtentsOut, _LayoutPolicyOut>>;
    constexpr auto __max_rank = ::cuda::std::max(_ExtentsIn::rank(), _ExtentsOut::rank());
    // Hybrid mdspan flow:
    //   1. Remove extent-1 modes.
    //   2. Optionally factorize the lower-rank tensor into an exact same-extents candidate.
    //   3. Exact same-extents candidates use the paired normalization path.
    //   4. Different-extents cases use independent sorting and common-prefix scoring,
    //      but only execute reordered copies when the copy collapses to one full tile.
    auto __src_raw = cudax::__to_raw_tensor<__extent_t, __stride_t, __max_rank>(__src, __remove_extent1);
    auto __dst_raw = cudax::__to_raw_tensor<__extent_t, __stride_t, __max_rank>(__dst, __remove_extent1);

    const bool __same_extents = cudax::__same_extents(__src_raw, __dst_raw);

    auto __src_candidate         = __src_raw;
    auto __dst_candidate         = __dst_raw;
    bool __factorization_success = false;

    if (__src_raw.__rank != __dst_raw.__rank)
    {
      if (__src_raw.__rank < __dst_raw.__rank)
      {
        __factorization_success = cudax::__try_factorize_to_match_sorted(__src_raw, __dst_sorted, __src_candidate);
        if (__factorization_success)
        {
          __dst_candidate = cudax::__erase_order(__dst_sorted);
        }
      }
      else if (__dst_raw.__rank < __src_raw.__rank)
      {
        __factorization_success = cudax::__try_factorize_to_match_sorted(__dst_raw, __src_sorted, __dst_candidate);
        if (__factorization_success)
        {
          __src_candidate = cudax::__erase_order(__src_sorted);
        }
      }
    }

    const bool __same_extent_candidate = __same_extents || __factorization_success;
    auto __src_normalized              = __src_raw;
    auto __dst_normalized              = __dst_raw;
    bool __use_pair_order              = false;
    size_t __normalized_tile_size      = 1;

    if (__same_extent_candidate)
    {
      __src_normalized = __src_candidate;
      __dst_normalized = __dst_candidate;
      cudax::__sort_by_stride_paired_generalized(__src_normalized, __dst_normalized);
      cudax::__flip_negative_strides_paired_generalized(__src_normalized, __dst_normalized);
      cudax::__coalesce_paired_generalized(__src_normalized, __dst_normalized);
      const bool __both_stride1 = (__src_normalized.__strides[0] == 1) && (__dst_normalized.__strides[0] == 1);
      __normalized_tile_size =
        __both_stride1 ? static_cast<size_t>(__src_normalized.__extents[0]) : static_cast<size_t>(1);
      __use_pair_order = (__normalized_tile_size > 1);
    }
    else
    {
      const size_t __uniform_rank     = ::cuda::std::max(__src_raw.__rank, __dst_raw.__rank);
      const auto __src_sorted         = cudax::__sort_by_stride(__src_raw);
      const auto __dst_sorted         = cudax::__sort_by_stride(__dst_raw);
      const auto __src_sorted_uniform = cudax::__append<__max_rank>(__src_sorted, __uniform_rank);
      const auto __dst_sorted_uniform = cudax::__append<__max_rank>(__dst_sorted, __uniform_rank);
      __normalized_tile_size          = cudax::__max_common_contiguous_size(__src_sorted_uniform, __dst_sorted_uniform);
    }
    const bool __full_tile_copy = (__normalized_tile_size == __tensor_size) && !__same_extent_candidate;

    const size_t __tile_size = (__full_tile_copy || __use_pair_order) ? __normalized_tile_size : 1;

    const size_t __num_tiles  = __tensor_size / __tile_size;
    const size_t __copy_bytes = __tile_size * sizeof(_TpIn);
    __tile_iterator_linearized<__extent_t, __stride_t, _TpIn, __max_rank> __src_tiles_iterator(
      __src_raw, __src_normalized, __tile_size, __use_pair_order);
    __tile_iterator_linearized<__extent_t, __stride_t, _TpOut, __max_rank> __dst_tiles_iterator(
      __dst_raw, __dst_normalized, __tile_size, __use_pair_order);
    cudax::__memcpy_batch_tiles(
      __src_tiles_iterator,
      __dst_tiles_iterator,
      __num_tiles,
      __copy_bytes,
      __direction,
      __src.data_handle(),
      __dst.data_handle(),
      __stream);
  }
}

/***********************************************************************************************************************
 * Public API
 **********************************************************************************************************************/

template <typename _TpIn,
          typename _ExtentsIn,
          typename _LayoutPolicyIn,
          typename _AccessorPolicyIn,
          typename _TpOut,
          typename _ExtentsOut,
          typename _LayoutPolicyOut,
          typename _AccessorPolicyOut>
_CCCL_HOST_API void copy_bytes(::cuda::host_mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn> __src,
                               ::cuda::device_mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut> __dst,
                               ::cuda::stream_ref __stream)
{
  using __src_type = ::cuda::std::mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn>;
  using __dst_type = ::cuda::std::mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut>;
  ::cuda::experimental::__copy_bytes_impl(
    static_cast<__src_type>(__src), static_cast<__dst_type>(__dst), __copy_direction::__host_to_device, __stream);
}

template <typename _TpIn,
          typename _ExtentsIn,
          typename _LayoutPolicyIn,
          typename _AccessorPolicyIn,
          typename _TpOut,
          typename _ExtentsOut,
          typename _LayoutPolicyOut,
          typename _AccessorPolicyOut>
_CCCL_HOST_API void copy_bytes(::cuda::device_mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn> __src,
                               ::cuda::host_mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut> __dst,
                               ::cuda::stream_ref __stream)
{
  using __src_type = ::cuda::std::mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn>;
  using __dst_type = ::cuda::std::mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut>;
  ::cuda::experimental::__copy_bytes_impl(
    static_cast<__src_type>(__src), static_cast<__dst_type>(__dst), __copy_direction::__device_to_host, __stream);
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_MDSPAN_D2H_H2D_H
