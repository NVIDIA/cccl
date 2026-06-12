//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___NCCL_ABI_COMPATIBLE_H
#define _CUDA_EXPERIMENTAL___NCCL_ABI_COMPATIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__type_traits/underlying_type.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__nccl::__abi_detail
{
template <class _Tp, class _Up>
[[nodiscard]] _CCCL_HOST_API constexpr bool __abi_compatible() noexcept;

template <class _R1, class... _Args1, class _R2, class... _Args2>
[[nodiscard]] _CCCL_HOST_API constexpr bool __abi_compatible_func(_R1 (*)(_Args1...), _R2 (*)(_Args2...)) noexcept
{
  if constexpr (::cuda::experimental::__nccl::__abi_detail::__abi_compatible<_R1, _R2>()
                && (sizeof...(_Args1) == sizeof...(_Args2)))
  {
    return (::cuda::experimental::__nccl::__abi_detail::__abi_compatible<_Args1, _Args2>() && ...);
  }
  return false;
}

template <class _Tp, class _Up>
[[nodiscard]] _CCCL_HOST_API constexpr bool __abi_compatible() noexcept
{
  using _RawTp = ::cuda::std::remove_cv_t<_Tp>;
  using _RawUp = ::cuda::std::remove_cv_t<_Up>;

  if constexpr (::cuda::std::is_same_v<_RawTp, _RawUp>)
  {
    // Equal types are obviously ABI compatible
    return true;
  }
  else if constexpr (::cuda::std::is_function_v<_RawTp> && ::cuda::std::is_function_v<_RawUp>)
  {
    // Functions need all arguments checked
    return ::cuda::experimental::__nccl::__abi_detail::__abi_compatible_func(
      ::cuda::std::decay_t<_RawTp>{}, ::cuda::std::decay_t<_RawUp>{});
  }
  else if constexpr (::cuda::std::is_enum_v<_RawTp> || ::cuda::std::is_enum_v<_RawUp>)
  {
    // If either side is an enum, we need to unwrap to check whether the underlying types
    // match. These must match *exactly*, otherwise we perform the moral equivalent of a
    // bitcast when we reinterpret them
    if constexpr (::cuda::std::is_enum_v<_RawTp> && ::cuda::std::is_enum_v<_RawUp>)
    {
      return ::cuda::experimental::__nccl::__abi_detail::__abi_compatible<::cuda::std::underlying_type_t<_RawTp>,
                                                                          ::cuda::std::underlying_type_t<_RawUp>>();
    }
    else if constexpr (::cuda::std::is_enum_v<_RawTp>)
    {
      return ::cuda::experimental::__nccl::__abi_detail::__abi_compatible<::cuda::std::underlying_type_t<_RawTp>,
                                                                          _RawUp>();
    }
    else
    {
      return ::cuda::experimental::__nccl::__abi_detail::__abi_compatible<_RawTp,
                                                                          ::cuda::std::underlying_type_t<_RawUp>>();
    }
  }
  else if constexpr (::cuda::std::is_pointer_v<_RawTp> && ::cuda::std::is_pointer_v<_RawUp>)
  {
    // Note the &&. If one is a pointer but the other is not, that's an error
    return ::cuda::experimental::__nccl::__abi_detail::__abi_compatible<::cuda::std::remove_pointer_t<_RawTp>,
                                                                        ::cuda::std::remove_pointer_t<_RawUp>>();
  }
  else
  {
    return false;
  }
}
} // namespace cuda::experimental::__nccl::__abi_detail

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___NCCL_ABI_COMPATIBLE_H
