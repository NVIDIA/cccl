//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_ARRIVAL_TOKEN_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_ARRIVAL_TOKEN_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/exchange.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <class _Tp, class _SynchronizerInstanceTag>
class __int_arrival_token
{
  _Tp __value_;

public:
  __int_arrival_token() = delete;

  _CCCL_DEVICE constexpr __int_arrival_token(_Tp __v) noexcept
      : __value_{__v}
  {}

  __int_arrival_token(const __int_arrival_token&) = delete;

  _CCCL_HIDE_FROM_ABI constexpr __int_arrival_token(__int_arrival_token&&) noexcept = default;

  __int_arrival_token& operator=(const __int_arrival_token&) = delete;

  _CCCL_HIDE_FROM_ABI constexpr __int_arrival_token& operator=(__int_arrival_token&&) noexcept = default;

  [[nodiscard]] _CCCL_DEVICE_API constexpr _Tp __value() const noexcept
  {
    return __value_;
  }
};

template <class _SynchronizerInstanceTag>
class __dummy_arrival_token
{
#  if defined(CCCL_ENABLE_DEVICE_ASSERTIONS)
  static constexpr ::cuda::std::uint32_t __valid_value   = 0xdead'beef;
  static constexpr ::cuda::std::uint32_t __invalid_value = 0xfeeb'daed;

  ::cuda::std::uint32_t __value_ = __valid_value;
#  endif // CCCL_ENABLE_DEVICE_ASSERTIONS

public:
  _CCCL_HIDE_FROM_ABI explicit constexpr __dummy_arrival_token() noexcept = default;

  __dummy_arrival_token(const __dummy_arrival_token&) = delete;

  __dummy_arrival_token& operator=(const __dummy_arrival_token&) = delete;

#  if defined(CCCL_ENABLE_DEVICE_ASSERTIONS)
  _CCCL_DEVICE_API constexpr __dummy_arrival_token(__dummy_arrival_token&& __other) noexcept
      : __value_{::cuda::std::exchange(__other.__value_, ::cuda::std::uint32_t{__invalid_value})}
  {}

  _CCCL_DEVICE_API constexpr __dummy_arrival_token& operator=(__dummy_arrival_token&& __other) noexcept
  {
    __value_ = ::cuda::std::exchange(__other.__value_, ::cuda::std::uint32_t{__invalid_value});
    return *this;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr bool __valid() const noexcept
  {
    return __value_ == ::cuda::std::uint32_t{__valid_value};
  }
#  else // ^^^ CCCL_ENABLE_DEVICE_ASSERTIONS ^^^ / vvv !CCCL_ENABLE_DEVICE_ASSERTIONS vvv
  _CCCL_HIDE_FROM_ABI constexpr __dummy_arrival_token(__dummy_arrival_token&&) noexcept = default;

  _CCCL_HIDE_FROM_ABI constexpr __dummy_arrival_token& operator=(__dummy_arrival_token&&) noexcept = default;

  [[nodiscard]] _CCCL_DEVICE_API constexpr bool __valid() const noexcept
  {
    return true;
  }
#  endif // ^^^ !CCCL_ENABLE_DEVICE_ASSERTIONS ^^^
};
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_ARRIVAL_TOKEN_CUH
