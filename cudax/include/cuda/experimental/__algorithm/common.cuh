//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ALGORITHM_COMMON
#define __CUDAX_ALGORITHM_COMMON

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/mdspan>
#include <cuda/std/span>

#include <cuda/experimental/__launch/launch_transform.cuh>

namespace cuda::experimental
{

template <typename _Tp>
_CCCL_CONCEPT __valid_1d_copy_fill_argument = _CUDA_VRANGES::contiguous_range<detail::__as_copy_arg_t<_Tp>>;

template <typename _Tp, typename _Decayed = _CUDA_VSTD::decay_t<_Tp>>
using __as_mdspan_t =
  _CUDA_VSTD::mdspan<typename _Decayed::value_type,
                     typename _Decayed::extents_type,
                     typename _Decayed::layout_type,
                     typename _Decayed::accessor_type>;

template <typename _Tp, typename = int>
inline constexpr bool __convertible_to_mdspan = false;

template <typename _Tp>
inline constexpr bool
  __convertible_to_mdspan<_Tp, _CUDA_VSTD::enable_if_t<_CUDA_VSTD::is_convertible_v<_Tp, __as_mdspan_t<_Tp>>, int>> =
    true;

template <typename _Tp>
inline constexpr bool __valid_nd_copy_fill_argument = __convertible_to_mdspan<detail::__as_copy_arg_t<_Tp>>;

} // namespace cuda::experimental
#endif //__CUDAX_ALGORITHM_COMMON
