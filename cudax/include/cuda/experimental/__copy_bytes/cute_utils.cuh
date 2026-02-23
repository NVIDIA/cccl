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

#include <cuda/experimental/__copy/types.cuh>

#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/tensor_impl.hpp>
//
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Create a CuTe tensor backed by global memory.
template <typename _Tp, typename _Layout>
[[nodiscard]] _CCCL_API ::cute::Tensor<_Tp, _Layout> __make_gmem_tensor(_Tp* __ptr, const _Layout& __layout) noexcept
{
  return ::cute::make_tensor(::cute::make_gmem_ptr(__ptr), __layout);
}

#if !_CCCL_COMPILER(NVRTC)

inline constexpr auto __remove_extent1_mode = ::cuda::std::true_type{};

//! @brief Construct a raw tensor from a pointer and a CuTe layout.
//!
//! Extracts runtime shapes and strides from the CuTe layout into a __raw_tensor.
//! For static layouts, compile-time values are converted to runtime integers.
//!
//! @tparam _MaxRank Maximum rank capacity for the raw tensor
//! @tparam _Tp      Element type (deduced from pointer)
//! @tparam _Layout  CuTe layout type
//! @param[in] __data   Pointer to tensor data
//! @param[in] __layout The CuTe layout to extract from
//! @return A __raw_tensor populated with the layout's shapes and strides
template <::cuda::std::size_t _MaxRank, typename _Tp, typename _Layout, bool _RemoveExtent1 = false>
[[nodiscard]] _CCCL_HOST_API __raw_tensor<_Tp, _MaxRank>
__to_raw_tensor(_Tp* __data, const _Layout& __layout, ::cuda::std::bool_constant<_RemoveExtent1> = {}) noexcept
{
  constexpr auto __rank = decltype(::cute::rank(__layout))::value;
  static_assert(__rank <= _MaxRank, "Layout rank exceeds maximum supported rank");
  __raw_tensor<_Tp, _MaxRank> __result{__data, 0, {}, {}};
  ::cuda::std::size_t __r = 0;
  ::cuda::static_for<__rank>([&] __host__ __device__(auto __i) {
    const auto __shape = ::cute::shape<__i>(__layout);
    if (!_RemoveExtent1 || __shape != 1)
    {
      __result.__shapes[__r]  = __shape;
      __result.__strides[__r] = ::cute::stride<__i>(__layout);
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
