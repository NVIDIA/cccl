//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_BYTES_CUTE_UTILS
#define _CUDAX__COPY_BYTES_CUTE_UTILS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/static_for.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/experimental/__copy/types.cuh>

#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/tensor_impl.hpp>
//
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Extracts the common scalar type from a CuTe shape or stride type.
//!
//! Recursively unwraps `cute::tuple<...>` and `cute::C<v>` to produce a single
//! arithmetic type suitable for `__raw_tensor` extent/stride arrays.
//!
//! - Scalar types (e.g., `int`, `int64_t`) pass through unchanged.
//! - `cute::C<v>` yields its `value_type` (e.g., `int` for `cute::Int<N>`).
//! - `cute::tuple<Ts...>` yields `common_type_t` of the recursively extracted types.
template <typename _Tuple>
struct __cute_common_type
{
  using type = _Tuple;
};

template <auto _V>
struct __cute_common_type<::cute::C<_V>>
{
  using type = typename ::cute::C<_V>::value_type;
};

template <typename... _Ts>
struct __cute_common_type<::cute::tuple<_Ts...>>
{
  using type = ::cuda::std::common_type_t<typename __cute_common_type<_Ts>::type...>;
};

template <typename _Tuple>
using __cute_common_type_t = typename __cute_common_type<_Tuple>::type;

//! @brief Create a CuTe tensor backed by global memory.
//!
//! @param[in] __ptr    Pointer to device memory
//! @param[in] __layout CuTe layout describing the tensor's shape and strides
//! @return A `cute::Tensor` with a `gmem_ptr` engine and the given layout
template <typename _Tp, typename _Layout>
[[nodiscard]] _CCCL_API auto __make_gmem_tensor(_Tp* __ptr, const _Layout& __layout) noexcept
{
  return ::cute::make_tensor(::cute::make_gmem_ptr(__ptr), __layout);
}

#if !_CCCL_COMPILER(NVRTC)

//! @brief Build a CuTe layout from a @ref __raw_tensor's shapes and strides.
//!
//! Mode 0 receives a compile-time `cute::_1{}` stride; remaining modes use the
//! runtime strides stored in the raw tensor. Call with
//! `make_index_sequence<N - 1>` where `N` is the desired rank.
//!
//! @param[in] __raw_tensor Raw tensor whose shapes/strides are extracted
//! @param[in] (tag) Index sequence of size `N - 1` (indices 0 .. N-2)
//! @return A CuTe layout of rank `N`
template <typename _Ep, typename _Sp, typename _Tp, ::cuda::std::size_t _MaxRank, ::cuda::std::size_t... _Is>
[[nodiscard]] _CCCL_HOST_API auto __to_cute_layout_contiguous(
  const __raw_tensor<_Ep, _Sp, _Tp, _MaxRank>& __raw_tensor, ::cuda::std::index_sequence<_Is...>) noexcept
{
  _CCCL_ASSERT(__raw_tensor.__strides[0] == 1, "First stride must be 1");
  return ::cute::make_layout(::cute::make_shape(__raw_tensor.__extents[0], __raw_tensor.__extents[_Is + 1]...),
                             ::cute::make_stride(::cute::_1{}, __raw_tensor.__strides[_Is + 1]...));
}

//! @brief Build a CuTe layout from a @ref __raw_tensor's shapes and strides.
//!
//! All modes use runtime strides (no compile-time stride assumptions).
//! Call with `make_index_sequence<N - 1>` where `N` is the desired rank.
//!
//! @param[in] __raw_tensor Raw tensor whose shapes/strides are extracted
//! @param[in] (tag) Index sequence of size `N - 1` (indices 0 .. N-2)
//! @return A CuTe layout of rank `N`
template <typename _Ep, typename _Sp, typename _Tp, ::cuda::std::size_t _MaxRank, ::cuda::std::size_t... _Is>
[[nodiscard]] _CCCL_HOST_API auto __to_cute_layout(const __raw_tensor<_Ep, _Sp, _Tp, _MaxRank>& __raw_tensor,
                                                   ::cuda::std::index_sequence<_Is...>) noexcept
{
  return ::cute::make_layout(::cute::make_shape(__raw_tensor.__extents[0], __raw_tensor.__extents[_Is + 1]...),
                             ::cute::make_stride(__raw_tensor.__strides[0], __raw_tensor.__strides[_Is + 1]...));
}

//! @brief Tag constant used to enable extent-1 mode removal in @ref __to_raw_tensor.
inline constexpr auto __remove_extent1_mode = ::cuda::std::true_type{};

//! @brief Construct a __raw_tensor from a pointer and a CuTe layout.
//!
//! Extracts runtime shapes and strides from the CuTe layout into a __raw_tensor.
//! The extent type `_Ep` and stride type `_Sp` are deduced from the layout's
//! `Shape` and `Stride` template parameters via @ref __cute_common_type.
//! When `_RemoveExtent1` is true, modes with shape == 1 are omitted and the
//! resulting rank may be less than the layout rank.
//!
//! @tparam _MaxRank       Maximum rank capacity for the raw tensor (must be >= layout rank)
//! @tparam _Tp            Element type (deduced from pointer)
//! @tparam _Shape         CuTe shape type (deduced from layout)
//! @tparam _Stride        CuTe stride type (deduced from layout)
//! @tparam _RemoveExtent1 If true, skip modes whose shape is 1
//! @param[in] __data   Pointer to tensor data
//! @param[in] __layout The CuTe layout to extract from
//! @return A __raw_tensor with rank <= layout rank, populated with the layout's shapes and strides
template <::cuda::std::size_t _MaxRank, typename _Tp, typename _Shape, typename _Stride, bool _RemoveExtent1 = false>
[[nodiscard]] _CCCL_HOST_API __raw_tensor<__cute_common_type_t<_Shape>, __cute_common_type_t<_Stride>, _Tp, _MaxRank>
__to_raw_tensor(_Tp* __data,
                const ::cute::Layout<_Shape, _Stride>& __layout,
                ::cuda::std::bool_constant<_RemoveExtent1> = {}) noexcept
{
  using _Ep             = __cute_common_type_t<_Shape>;
  using _Sp             = __cute_common_type_t<_Stride>;
  constexpr auto __rank = decltype(::cute::rank(__layout))::value;
  static_assert(__rank <= _MaxRank, "Layout rank exceeds maximum supported rank");
  __raw_tensor<_Ep, _Sp, _Tp, _MaxRank> __result{__data, 0, {}, {}};
  ::cuda::std::size_t __r = 0;
  ::cuda::static_for<__rank>([&] __host__ __device__(auto __i) {
    const auto __shape = ::cute::shape<__i>(__layout);
    if (!_RemoveExtent1 || __shape != 1)
    {
      __result.__extents[__r] = static_cast<_Ep>(__shape);
      __result.__strides[__r] = static_cast<_Sp>(::cute::stride<__i>(__layout));
      ++__r;
    }
  });
  __result.__rank = __r;
  return __result;
}

#endif // !_CCCL_COMPILER(NVRTC)
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_CUTE_UTILS
