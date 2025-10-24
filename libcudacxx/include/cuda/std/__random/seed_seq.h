//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANDOM_SEED_SEQ_H
#define _CUDA_STD___RANDOM_SEED_SEQ_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__algorithm/fill.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/initializer_list>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

class _CCCL_TYPE_VISIBILITY_DEFAULT seed_seq
{
public:
  // types
  using result_type = ::cuda::std::uint_least32_t;

  // constructors
  seed_seq() = default;

  template <class _InputIt>
  _CCCL_API constexpr seed_seq(_InputIt __begin, _InputIt __end)
  {
    auto __n = ::cuda::std::distance(__begin, __end);
    if (__n == 0)
    {
      return;
    }
    auto* m_ptr = new result_type[__n];
    for (auto __i = 0; __i < __n; ++__i, ++__begin)
    {
      auto val   = *__begin;
      m_ptr[__i] = static_cast<result_type>(val) & 0xFFFFFFFF;
    }
    __v_ = ::cuda::std::span<result_type>(m_ptr, __n);
  }

  template <typename _InitT,
            typename = typename ::cuda::std::enable_if<::cuda::std::is_convertible<_InitT, result_type>::value>::type>
  _CCCL_API constexpr seed_seq(std::initializer_list<_InitT> __il)
      : seed_seq(__il.begin(), __il.end())
  {}

  seed_seq(const seed_seq&) = delete;

// Constexpr destructor from C++20 onwards
#if _CCCL_STD_VER > 2017
  constexpr
#endif // _CCCL_STD_VER > 2017
    _CCCL_API ~seed_seq()
  {
    // cudamanagedfree
    if (!__v_.empty())
    {
      delete[] __v_.data();
    }
  }

  _CCCL_API auto operator=(const seed_seq&) = delete;
  _CCCL_API constexpr ::cuda::std::size_t size() const noexcept
  {
    return __v_.size();
  }

  template <class _OutputIt>
  _CCCL_API constexpr void param(_OutputIt __dest) const
  {
    ::cuda::std::copy(__v_.begin(), __v_.end(), __dest);
  }

  template <class _RandomIt>
  _CCCL_API constexpr void generate(_RandomIt __begin, _RandomIt __end) const
  {
    if (__begin == __end)
    {
      return;
    }
    // https://en.cppreference.com/w/cpp/numeric/random/seed_seq/generate.html
    const result_type __z = __v_.size();
    const result_type __n = ::cuda::std::distance(__begin, __end);
    const result_type __m = ::cuda::std::max(__z + 1, __n);
    const result_type __t = (__n >= 623) ? 11 : (__n >= 68) ? 7 : (__n >= 39) ? 5 : (__n >= 7) ? 3 : (__n - 1) / 2;
    const result_type __p = (__n - __t) / 2;
    const result_type __q = __p + __t;

    // 1.
    ::cuda::std::fill(__begin, __end, result_type{0x8b8b8b8b});

    // 2.
    for (::cuda::std::size_t __k = 0; __k < __m; ++__k)
    {
      result_type __k_mod_n   = __k % __n;
      result_type __k_p_mod_n = (__k + __p) % __n;
      result_type __k_q_mod_n = (__k + __q) % __n;
      // 2.1
      const result_type __r1 = 1664525 * __T(__begin[__k_mod_n] ^ __begin[__k_p_mod_n] ^ __begin[(__k - 1) % __n]);
      // 2.2
      result_type __r2 = (__k == 0) ? __r1 + __z : (__k <= __z) ? __r1 + __k_mod_n + __v_[__k - 1] : __r1 + __k_mod_n;
      // 2.3
      __begin[__k_p_mod_n] += __r1;
      __begin[__k_p_mod_n] &= 0xFFFFFFFF;
      __begin[__k_q_mod_n] += __r2;
      __begin[__k_q_mod_n] &= 0xFFFFFFFF;
      __begin[__k_mod_n] = __r2 & 0xFFFFFFFF;
    }

    // 3.
    for (::cuda::std::size_t __k = __m; __k < __m + __n; ++__k)
    {
      result_type __k_mod_n   = __k % __n;
      result_type __k_p_mod_n = (__k + __p) % __n;
      result_type __k_q_mod_n = (__k + __q) % __n;
      // 3.1
      const result_type __r3 = 1566083941 * __T(__begin[__k_mod_n] + __begin[__k_p_mod_n] + __begin[(__k - 1) % __n]);
      // 3.2
      const result_type __r4 = __r3 - __k_mod_n;
      // 3.3
      __begin[__k_p_mod_n] ^= __r3;
      __begin[__k_p_mod_n] &= 0xFFFFFFFF;
      // 3.4
      __begin[__k_q_mod_n] ^= __r4;
      __begin[__k_q_mod_n] &= 0xFFFFFFFF;
      // 3.5
      __begin[__k_mod_n] = __r4 & 0xFFFFFFFF;
    }
  }

private:
  _CCCL_API static constexpr result_type __T(result_type __x) noexcept
  {
    return (__x ^ (__x >> 27));
  }
  ::cuda::std::span<result_type> __v_{};
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_PHILOX_ENGINE_H
