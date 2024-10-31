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
#include <cuda/std/span>

#include <cuda/experimental/__launch/launch_transform.cuh>

namespace cuda::experimental
{
#if _CCCL_STD_VER >= 2020 && defined(_CCCL_SPAN_USES_RANGES)
template <typename _Tp>
concept __valid_copy_fill_argument = _CUDA_VRANGES::contiguous_range<detail::__as_copy_arg_t<_Tp>>;

#else
template <typename _Tp, typename = int>
inline constexpr bool __convertible_to_span = false;

template <typename _Tp>
inline constexpr bool __convertible_to_span<
  _Tp,
  _CUDA_VSTD::enable_if_t<
    _CUDA_VSTD::is_convertible_v<_Tp, _CUDA_VSTD::span<typename _CUDA_VSTD::decay_t<_Tp>::value_type>>,
    int>> = true;

template <typename _Tp>
inline constexpr bool __valid_copy_fill_argument =
  _CUDA_VRANGES::contiguous_range<detail::__as_copy_arg_t<_Tp>> || __convertible_to_span<_Tp>;

#endif

} // namespace cuda::experimental
#endif //__CUDAX_ALGORITHM_COMMON
