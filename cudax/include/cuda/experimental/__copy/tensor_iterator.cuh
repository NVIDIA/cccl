//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_TENSOR_ITERATOR_H
#define _CUDAX__COPY_TENSOR_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/fast_modulo_division.h>
#include <cuda/__utility/in_range.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
/***********************************************************************************************************************
 * Fast Modulo/Division based on Precomputation
 **********************************************************************************************************************/

template <typename _ExtentT, ::cuda::std::size_t _Size, ::cuda::std::size_t... _Rp>
[[nodiscard]] _CCCL_API ::cuda::std::array<::cuda::fast_mod_div<_ExtentT>, sizeof...(_Rp)> __extents_fast_div_mod_impl(
  const ::cuda::std::array<_ExtentT, _Size>& __extents, ::cuda::std::index_sequence<_Rp...> = {}) noexcept
{
  using __fast_mod_div_t = ::cuda::fast_mod_div<_ExtentT>;
  using __array_t        = ::cuda::std::array<__fast_mod_div_t, sizeof...(_Rp)>;
  return __array_t{__fast_mod_div_t(__extents[_Rp])...};
}

//! @brief Precompute modulo/division for each array extent.
//!
//! @param[in] __extents Array of extents
//! @return Array of precomputed fast modulo/division objects
template <typename _ExtentT, ::cuda::std::size_t _Size>
[[nodiscard]] _CCCL_API ::cuda::std::array<::cuda::fast_mod_div<_ExtentT>, _Size>
__extents_fast_div_mod(const ::cuda::std::array<_ExtentT, _Size>& __extents) noexcept
{
  using __seq_t = ::cuda::std::make_index_sequence<_Size>;
  return ::cuda::experimental::__extents_fast_div_mod_impl(__extents, __seq_t{});
}

//! @brief Compute the product of extents in the range [__start, __end).
//!
//! @param[in] __extents Array of extents
//! @param[in] __start   Start index of the range (inclusive)
//! @param[in] __end     End index of the range (exclusive)
//! @return Fast modulo/division object wrapping the product
template <typename _ExtentT, ::cuda::std::size_t _Size>
[[nodiscard]] _CCCL_API ::cuda::fast_mod_div<_ExtentT> __extent_product_in_range(
  const ::cuda::std::array<_ExtentT, _Size>& __extents, ::cuda::std::size_t __start, ::cuda::std::size_t __end) noexcept
{
  using __fast_mod_div_t = ::cuda::fast_mod_div<_ExtentT>;
  _CCCL_ASSERT(::cuda::in_range(__start, ::cuda::std::size_t{0}, _Size), "invalid __start");
  _CCCL_ASSERT(::cuda::in_range(__end, ::cuda::std::size_t{0}, _Size), "invalid __end");
  _ExtentT __size = 1;
  for (auto __i = __start; __i < __end; __i++)
  {
    __size *= __extents[__i];
  }
  return __fast_mod_div_t(__size);
}

template <typename _ExtentT, ::cuda::std::size_t _Size, ::cuda::std::size_t... _Rp>
[[nodiscard]] _CCCL_API ::cuda::std::array<::cuda::fast_mod_div<_ExtentT>, sizeof...(_Rp)>
__extents_product_fast_div_mod_impl(const ::cuda::std::array<_ExtentT, _Size>& __extents,
                                    ::cuda::std::index_sequence<_Rp...> = {}) noexcept
{
  using __fast_mod_div_t = ::cuda::fast_mod_div<_ExtentT>;
  using __array_t        = ::cuda::std::array<__fast_mod_div_t, sizeof...(_Rp)>;
  return __array_t{::cuda::experimental::__extent_product_in_range(__extents, 0, _Rp)...};
}

//! @brief Precompute cumulative extent products as fast modulo/division objects.
//!
//! @param[in] __extents Array of extents
//! @return Array of fast modulo/division objects for each cumulative product
template <typename _ExtentT, ::cuda::std::size_t _Size>
[[nodiscard]] _CCCL_API ::cuda::std::array<::cuda::fast_mod_div<_ExtentT>, _Size>
__extents_product_fast_div_mod(const ::cuda::std::array<_ExtentT, _Size>& __extents) noexcept
{
  using __seq_t = ::cuda::std::make_index_sequence<_Size>;
  return ::cuda::experimental::__extents_product_fast_div_mod_impl(__extents, __seq_t{});
}

/***********************************************************************************************************************
 * Tensor Coordinate Iterator and Partial Tensor
 **********************************************************************************************************************/

//! @brief Iterator that maps a linear tile index to a pointer into a strided raw tensor.
template <typename _ExtentT, ::cuda::std::size_t _Rank>
struct __tensor_coord_iterator
{
  using __fast_mod_div_t = ::cuda::fast_mod_div<_ExtentT>;
  using __array_t        = ::cuda::std::array<__fast_mod_div_t, _Rank>;

  __array_t __extents_;
  __array_t __extent_products_;

  //! @brief Convert an array of _UExtentT elements to an array of _ExtentT elements.
  //!
  //! @param[in] __in_array Source array with elements of type _UExtentT
  //! @return Array with elements statically cast to _ExtentT
  template <typename _UExtentT>
  [[nodiscard]] static _CCCL_HOST_API ::cuda::std::array<_ExtentT, _Rank>
  __to_extent_array(const ::cuda::std::array<_UExtentT, _Rank>& __in_array) noexcept
  {
    ::cuda::std::array<_ExtentT, _Rank> __out_array{};
    for (::cuda::std::size_t __i = 0; __i < _Rank; ++__i)
    {
      __out_array[__i] = static_cast<_ExtentT>(__in_array[__i]);
    }
    return __out_array;
  }

  //! @brief Constructs the iterator from tensor extents.
  //!
  //! @param[in] __extents Tensor extents (may be unsigned; converted to _ExtentT internally)
  template <typename _UExtentT>
  _CCCL_HOST_API explicit __tensor_coord_iterator(const ::cuda::std::array<_UExtentT, _Rank>& __extents) noexcept
      : __extents_{::cuda::experimental::__extents_fast_div_mod(__to_extent_array(__extents))}
      , __extent_products_{::cuda::experimental::__extents_product_fast_div_mod(__to_extent_array(__extents))}
  {}

  //! @brief Returns the multi-dimensional coordinates for the given linear index.
  //!
  //! @param[in] __in_arraydex Linear tile index
  //! @return Array of coordinates into the tensor
  [[nodiscard]] _CCCL_API ::cuda::std::array<_ExtentT, _Rank> operator()(_ExtentT __in_arraydex) const noexcept
  {
    if constexpr (_Rank == 1)
    {
      return ::cuda::std::array<_ExtentT, _Rank>{{__in_arraydex}};
    }
    else
    {
      ::cuda::std::array<_ExtentT, _Rank> __coords{};
      __coords[0] = __in_arraydex % __extents_[0]; // __extent_products_[0] == 1
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __i = 1; __i < _Rank; ++__i)
      {
        __coords[__i] = (__in_arraydex / __extent_products_[__i]) % __extents_[__i];
      }
      return __coords;
    }
  }
};

//! @brief Lightweight device-side wrapper providing coordinate-indexed access to strided tensor data.
//!
//! Wraps a data pointer, per-dimension strides, and an accessor into a callable that maps
//! multi-dimensional coordinates to element references.
template <typename _Tp, typename _StrideT, ::cuda::std::size_t _Rank, typename _Accessor>
struct __partial_tensor
{
  _Tp* __ptr;
  ::cuda::std::array<_StrideT, _Rank> __strides;
  _Accessor __accessor;

  //! @brief Access the element at the given multi-dimensional coordinates.
  //!
  //! @param[in] __coords Array of per-dimension coordinates
  //! @return Reference to the element at the computed offset
  template <typename _CoordT>
  [[nodiscard]] _CCCL_DEVICE_API decltype(auto)
  operator()(const ::cuda::std::array<_CoordT, _Rank>& __coords) const noexcept
  {
    _StrideT __offset = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < int{_Rank}; ++__i)
    {
      __offset += static_cast<_StrideT>(__coords[__i]) * __strides[__i];
    }
    return __accessor.access(const_cast<::cuda::std::remove_const_t<_Tp>*>(__ptr), __offset);
  }
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_TENSOR_ITERATOR_H
