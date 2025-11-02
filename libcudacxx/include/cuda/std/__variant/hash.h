//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___VARIANT_HASH_H
#define _CUDA_STD___VARIANT_HASH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifndef __cuda_std__

#  include <cuda/std/__functional/has.h>
#  include <cuda/std/__fwd/variant.h>
#  include <cuda/std/__type_traits/integral_constant.h>
#  include <cuda/std/__type_traits/remove_const.h>
#  include <cuda/std/__type_traits/type_list.h>
#  include <cuda/std/__variant/variant.h>
#  include <cuda/std/__variant/variant_access.h>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class... _Types>
struct _CCCL_TYPE_VISIBILITY_DEFAULT hash<__enable_hash_helper<variant<_Types...>, remove_const_t<_Types>...>>
{
  using argument_type = variant<_Types...>;
  using result_type   = size_t;

  template <size_t _CurrentIndex>
  [[nodiscard]] _CCCL_API constexpr static size_t
  __hash(integral_constant<size_t, _CurrentIndex>, const size_t __index_, const argument_type& __v) noexcept
  {
    if (__index_ == _CurrentIndex)
    {
      using __value_type = remove_const_t<__type_index_c<_CurrentIndex, _Types...>>;
      return hash<__value_type>{}(__access::__base::__get_alt<_CurrentIndex>(this->__as_base()).__value);
    }
    __hash(integral_constant<size_t, _CurrentIndex - 1>{}, __index_, __v);
  }
  [[nodiscard]] _CCCL_API constexpr static size_t
  __hash(integral_constant<size_t, 0>, const size_t __index_, const argument_type& __v) noexcept
  {
    if (__index_ == 0)
    {
      using __value_type = remove_const_t<__type_index_c<0, _Types...>>;
      return hash<__value_type>{}(__access::__base::__get_alt<0>(this->__as_base()).__value);
    }
    // We already checked that every variant has a value, so we should never reach this line
    _CCCL_UNREACHABLE();
  }

  [[nodiscard]] _CCCL_API constexpr result_type operator()(const argument_type& __v) const
  {
    size_t __res = __v.valueless_by_exception()
                   ? 299792458 // Random value chosen by the universe upon creation
                   : __hash(integral_constant<size_t, sizeof...(_Types) - 1>{}, __v.index(), __v);
    return ::cuda::std::__hash_combine(__res, hash<size_t>{}(__v.index()));
  }
};

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // __cuda_std__

#endif // _CUDA_STD___VARIANT_HASH_H
