//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___RANDOM_RANDOM_BIJECTION_H
#define _CUDA___RANDOM_RANDOM_BIJECTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__random/feistel_bijection.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Adaptor for a __bijection to work with any size problem. It achieves this by iterating the __bijection until
//! the result is less than __num_elements. For a feistel bijection, the worst case number of iterations required for
//! one call to operator() is O(__num_elements) with low probability. It has amortised O(1) complexity.
//! @tparam _IndexType The type of the index to shuffle. Defaults to uint64_t.
//! @tparam _Bijection The __bijection to use. A low quality random __bijection may lead to poor work balancing between
//! calls to the operator(). Defaults to a feistel bijetion
template <class _IndexType = ::cuda::std::uint64_t, class _Bijection = __feistel_bijection>
class random_bijection
{
private:
  static_assert(::cuda::std::is_integral_v<_IndexType>, "_IndexType must be an integral type");
  static_assert(::cuda::std::is_integral_v<typename _Bijection::index_type>,
                "_Bijection::index_type must be an integral type");
  static_assert(::cuda::std::is_convertible_v<_IndexType, typename _Bijection::index_type>,
                "_IndexType must be convertible to _Bijection::index_type");

  _Bijection __bijection{};
  _IndexType __num_elements{};

public:
  using index_type = _IndexType;

  _CCCL_HIDE_FROM_ABI constexpr random_bijection() noexcept = default;

  template <class _RNG>
  _CCCL_API constexpr random_bijection(_IndexType __num_elements, _RNG&& __gen) noexcept
      : __bijection(__num_elements, ::cuda::std::forward<_RNG>(__gen))
      , __num_elements(__num_elements)
  {}

  [[nodiscard]] _CCCL_API constexpr _IndexType operator()(_IndexType __n) const noexcept
  {
    // The initial index must be be in the range [0, __num_elemments]
    // If __n < __num_elements Iterating a __bijection like this will always terminate.
    // If __n >= __num_elements, then this may loop forever.
    _CCCL_ASSERT(__n < __num_elements, "random_bijection::operator(): index out of range");
    do
    {
      __n = static_cast<_IndexType>(__bijection(__n));
    } while (__n >= __num_elements);

    return __n;
  }

  [[nodiscard]] _CCCL_API constexpr _IndexType size() const noexcept
  {
    return __num_elements;
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___RANDOM_RANDOM_BIJECTION_H
