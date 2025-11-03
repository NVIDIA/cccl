//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TUPLE_GET_H
#define _CUDA_STD___TUPLE_GET_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/get.h>
#include <cuda/std/__tuple_dir/tuple.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <size_t _Ip, class... _Tp>
[[nodiscard]] _CCCL_API constexpr tuple_element_t<_Ip, tuple<_Tp...>>& get(tuple<_Tp...>& __t) noexcept
{
  return __t.template __get_impl<_Ip>();
}

template <size_t _Ip, class... _Tp>
[[nodiscard]] _CCCL_API constexpr const tuple_element_t<_Ip, tuple<_Tp...>>& get(const tuple<_Tp...>& __t) noexcept
{
  return __t.template __get_impl<_Ip>();
}

template <size_t _Ip, class... _Tp>
[[nodiscard]] _CCCL_API constexpr tuple_element_t<_Ip, tuple<_Tp...>>&& get(tuple<_Tp...>&& __t) noexcept
{
  return ::cuda::std::move(__t).template __get_impl<_Ip>();
}

template <size_t _Ip, class... _Tp>
[[nodiscard]] _CCCL_API constexpr const tuple_element_t<_Ip, tuple<_Tp...>>&& get(const tuple<_Tp...>&& __t) noexcept
{
  return ::cuda::std::move(__t).template __get_impl<_Ip>();
}

namespace __find_detail
{
static constexpr size_t __not_found = ~size_t(0);
static constexpr size_t __ambiguous = __not_found - 1;

[[nodiscard]] _CCCL_API constexpr size_t __find_idx_return(size_t __curr_i, size_t __res, bool __matches) noexcept
{
  return !__matches ? __res : (__res == __not_found ? __curr_i : __ambiguous);
}

template <size_t _Nx>
[[nodiscard]] _CCCL_API constexpr size_t __find_idx(size_t __i, const bool (&__matches)[_Nx]) noexcept
{
  return __i == _Nx ? __not_found : __find_idx_return(__i, __find_idx(__i + 1, __matches), __matches[__i]);
}

template <class _T1, class... _Args>
struct __find_exactly_one_checked
{
  static constexpr bool __matches[sizeof...(_Args)] = {is_same_v<_T1, _Args>...};
  static constexpr size_t value                     = __find_detail::__find_idx(0, __matches);
  static_assert(value != __not_found, "type not found in type list");
  static_assert(value != __ambiguous, "type occurs more than once in type list");
};

template <class _T1>
struct __find_exactly_one_checked<_T1>
{
  static_assert(!is_same_v<_T1, _T1>, "type not in empty type list");
};
} // namespace __find_detail

template <typename _T1, typename... _Args>
struct __find_exactly_one_t : public __find_detail::__find_exactly_one_checked<_T1, _Args...>
{};

template <class _T1, class... _Args>
[[nodiscard]] _CCCL_API constexpr _T1& get(tuple<_Args...>& __tup) noexcept
{
  return ::cuda::std::get<__find_exactly_one_t<_T1, _Args...>::value>(__tup);
}

template <class _T1, class... _Args>
[[nodiscard]] _CCCL_API constexpr _T1 const& get(tuple<_Args...> const& __tup) noexcept
{
  return ::cuda::std::get<__find_exactly_one_t<_T1, _Args...>::value>(__tup);
}

template <class _T1, class... _Args>
[[nodiscard]] _CCCL_API constexpr _T1&& get(tuple<_Args...>&& __tup) noexcept
{
  return ::cuda::std::get<__find_exactly_one_t<_T1, _Args...>::value>(::cuda::std::move(__tup));
}

template <class _T1, class... _Args>
[[nodiscard]] _CCCL_API constexpr _T1 const&& get(tuple<_Args...> const&& __tup) noexcept
{
  return ::cuda::std::get<__find_exactly_one_t<_T1, _Args...>::value>(::cuda::std::move(__tup));
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_GET_H
