//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_PROBING_SCHEME_CUH
#define _CUDAX___CUCO_PROBING_SCHEME_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <cuda/experimental/__cuco/__detail/probing_scheme_base.cuh>
#include <cuda/experimental/__cuco/traits.hpp>

#include <cooperative_groups.h>

namespace cuda::experimental::cuco
{
//! @brief Public linear probing scheme class.
//!
//! Template parameters:
//! - `_CgSize`: Cooperative group size
//! - `_Hash`: Hash functor type
template <int _CgSize, class _Hash>
class linear_probing : private ::cuda::experimental::cuco::__detail::__probing_scheme_base<_CgSize>
{
  using __base_type = ::cuda::experimental::cuco::__detail::__probing_scheme_base<_CgSize>;

public:
  static constexpr int cg_size = __base_type::__cg_size;
  using hasher                 = _Hash;

  _CCCL_API constexpr linear_probing(const _Hash& __hash = {})
      : __hash{__hash}
  {}

  template <class _NewHash>
  [[nodiscard]] _CCCL_API constexpr auto rebind_hash_function(const _NewHash& __hash) const noexcept
  {
    return linear_probing<cg_size, _NewHash>{__hash};
  }

  template <int _BucketSize, class _ProbeKey, class _Extent>
  _CCCL_API constexpr auto make_iterator(_ProbeKey __probe_key, _Extent __upper_bound) const noexcept;

  template <int _BucketSize, class _ProbeKey, class _Extent, class _ParentCG>
  _CCCL_API constexpr auto make_iterator(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group,
                                         _ProbeKey __probe_key,
                                         _Extent __upper_bound) const noexcept;

  _CCCL_API constexpr hasher hash_function() const noexcept
  {
    return __hash;
  }

private:
  _Hash __hash;
};

//! @brief Public double hashing scheme class.
//!
//! Template parameters:
//! - `_CgSize`: Cooperative group size
//! - `_Hash1`: First hash functor
//! - `_Hash2`: Second hash functor
template <int _CgSize, class _Hash1, class _Hash2 = _Hash1>
class double_hashing : private ::cuda::experimental::cuco::__detail::__probing_scheme_base<_CgSize>
{
  using __base_type = ::cuda::experimental::cuco::__detail::__probing_scheme_base<_CgSize>;

public:
  static constexpr int cg_size = __base_type::__cg_size;
  using hasher                 = ::cuda::std::tuple<_Hash1, _Hash2>;

  _CCCL_API constexpr double_hashing(const _Hash1& __hash1 = {}, const _Hash2& __hash2 = {1})
      : __hash1{__hash1}
      , __hash2{__hash2}
  {}

  _CCCL_API constexpr double_hashing(const ::cuda::std::tuple<_Hash1, _Hash2>& __hash)
      : __hash1{__hash.first}
      , __hash2{__hash.second}
  {}

  template <class _NewHash,
            class _Enable = ::cuda::std::enable_if_t<::cuda::experimental::cuco::is_tuple_like<_NewHash>::value>>
  [[nodiscard]] _CCCL_API constexpr auto rebind_hash_function(const _NewHash& __hash) const
  {
    static_assert(::cuda::experimental::cuco::is_tuple_like<_NewHash>::value,
                  "The given hasher must be a tuple-like object");

    auto const [__hash1, __hash2] = ::cuda::std::tuple{__hash};
    using __hash1_type            = ::cuda::std::decay_t<decltype(__hash1)>;
    using __hash2_type            = ::cuda::std::decay_t<decltype(__hash2)>;
    return double_hashing<cg_size, __hash1_type, __hash2_type>{__hash1, __hash2};
  }

  template <int _BucketSize, class _ProbeKey, class _Extent>
  _CCCL_API constexpr auto make_iterator(_ProbeKey __probe_key, _Extent __upper_bound) const noexcept;

  template <int _BucketSize, class _ProbeKey, class _Extent, class _ParentCG>
  _CCCL_API constexpr auto make_iterator(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group,
                                         _ProbeKey __probe_key,
                                         _Extent __upper_bound) const noexcept;

  _CCCL_API constexpr hasher hash_function() const noexcept
  {
    return {__hash1, __hash2};
  }

private:
  _Hash1 __hash1;
  _Hash2 __hash2;
};

//! @brief Trait indicating whether a probing scheme is double hashing.
template <class _Tp>
struct is_double_hashing : ::cuda::std::false_type
{};

template <int _CgSize, class _Hash1, class _Hash2>
struct is_double_hashing<double_hashing<_CgSize, _Hash1, _Hash2>> : ::cuda::std::true_type
{};
} // namespace cuda::experimental::cuco

#include <cuda/experimental/__cuco/__detail/probing_scheme_impl.cuh>

#endif // _CUDAX___CUCO_PROBING_SCHEME_CUH
