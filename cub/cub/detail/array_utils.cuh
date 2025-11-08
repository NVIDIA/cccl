// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/type_traits.cuh> // static_size_v
#include <cub/util_namespace.cuh>

#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>

CUB_NAMESPACE_BEGIN
namespace detail
{
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

/***********************************************************************************************************************
 * Generic Array-like to Array Conversion
 **********************************************************************************************************************/

template <typename CastType, typename Input, ::cuda::std::size_t... i>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::array<CastType, static_size_v<Input>>
to_array_impl(const Input& input, ::cuda::std::index_sequence<i...>)
{
  using ArrayType = ::cuda::std::array<CastType, static_size_v<Input>>;
  return ArrayType{static_cast<CastType>(input[i])...};
}

template <typename CastType = void, typename Input>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::array<CastType, static_size_v<Input>>
to_array(const Input& input)
{
  using InputType = ::cuda::std::iter_value_t<Input>;
  using CastType1 = ::cuda::std::_If<::cuda::std::is_same_v<CastType, void>, InputType, CastType>;
  return to_array_impl<CastType1>(input, ::cuda::std::make_index_sequence<static_size_v<Input>>{});
}

#endif // !_CCCL_DOXYGEN_INVOKED
} // namespace detail
CUB_NAMESPACE_END
