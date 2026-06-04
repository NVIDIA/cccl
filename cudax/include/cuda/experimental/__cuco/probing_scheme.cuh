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
//! @note Linear probing is efficient when few collisions are present, e.g., low occupancy or low
//! multiplicity.
//!
//! @note `_Hash` should be a callable object type.
//!
//! @tparam _CgSize Cooperative group size
//! @tparam _Hash Hash functor type
template <int _CgSize, class _Hash>
class linear_probing : private ::cuda::experimental::cuco::__detail::__probing_scheme_base<_CgSize>
{
  using __base_type = ::cuda::experimental::cuco::__detail::__probing_scheme_base<_CgSize>;

public:
  static constexpr int cg_size = __base_type::__cg_size;
  using hasher                 = _Hash;

  //! @brief Constructs a linear probing scheme with the given hasher callable.
  //!
  //! @param __hash Hasher
  _CCCL_HOST_DEVICE_API constexpr linear_probing(const _Hash& __hash = {})
      : __hash{__hash}
  {}

  //! @brief Makes a copy of the current probing scheme with the given hasher.
  //!
  //! @tparam _NewHash New hasher type
  //!
  //! @param __hash Hasher
  //!
  //! @return Copy of the current probing scheme
  template <class _NewHash>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto rebind_hash_function(const _NewHash& __hash) const noexcept
  {
    return linear_probing<cg_size, _NewHash>{__hash};
  }

  //! @brief Returns a probing iterator.
  //!
  //! @tparam _BucketSize Size of the bucket
  //! @tparam _ProbeKey Type of probing key
  //! @tparam _Extent Type of extent
  //!
  //! @param __probe_key The probing key
  //! @param __upper_bound Upper bound of the iteration
  //!
  //! @return An iterator whose value_type is convertible to the slot index type
  template <int _BucketSize, class _ProbeKey, class _Extent>
  _CCCL_HOST_DEVICE_API constexpr auto make_iterator(_ProbeKey __probe_key, _Extent __upper_bound) const noexcept
  {
    using __size_type        = typename _Extent::index_type;
    __size_type const __init = __hash(__probe_key) % (__upper_bound.extent(0) / _BucketSize) * _BucketSize;
    return ::cuda::experimental::cuco::__detail::__probing_iterator<_Extent>{
      __init, static_cast<__size_type>(_BucketSize), __upper_bound};
  }

  //! @brief Returns a cooperative group based probing iterator.
  //!
  //! @tparam _BucketSize Size of the bucket
  //! @tparam _ProbeKey Type of probing key
  //! @tparam _Extent Type of extent
  //! @tparam _ParentCG Type of parent cooperative group
  //!
  //! @param __group The cooperative group used to generate the probing iterator
  //! @param __probe_key The probing key
  //! @param __upper_bound Upper bound of the iteration
  //!
  //! @return An iterator whose value_type is convertible to the slot index type
  template <int _BucketSize, class _ProbeKey, class _Extent, class _ParentCG>
  _CCCL_HOST_DEVICE_API constexpr auto
  make_iterator(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group,
                _ProbeKey __probe_key,
                _Extent __upper_bound) const noexcept
  {
    using __size_type              = typename _Extent::index_type;
    constexpr __size_type __stride = cg_size * _BucketSize;
    __size_type const __init       = __hash(__probe_key) % (__upper_bound.extent(0) / __stride) * __stride
                             + static_cast<__size_type>(__group.thread_rank() * _BucketSize);
    return ::cuda::experimental::cuco::__detail::__probing_iterator<_Extent>{__init, __stride, __upper_bound};
  }

  //! @brief Gets the function used to hash keys.
  //!
  //! @return The function used to hash keys
  _CCCL_HOST_DEVICE_API constexpr hasher hash_function() const noexcept
  {
    return __hash;
  }

private:
  _Hash __hash;
};

//! @brief Public double hashing scheme class.
//!
//! @note Default probing scheme for cuco data structures. It shows superior performance over linear
//! probing especially when dealing with high multiplicity and/or high occupancy use cases.
//!
//! @note `_Hash1` and `_Hash2` should be callable object types.
//!
//! @note `_Hash2` needs to be able to construct from an integer value to avoid secondary clustering.
//!
//! @tparam _CgSize Cooperative group size
//! @tparam _Hash1 First hash functor
//! @tparam _Hash2 Second hash functor
template <int _CgSize, class _Hash1, class _Hash2 = _Hash1>
class double_hashing : private ::cuda::experimental::cuco::__detail::__probing_scheme_base<_CgSize>
{
  using __base_type = ::cuda::experimental::cuco::__detail::__probing_scheme_base<_CgSize>;

public:
  static constexpr int cg_size = __base_type::__cg_size;
  using hasher                 = ::cuda::std::tuple<_Hash1, _Hash2>;

  //! @brief Constructs a double hashing probing scheme with the two hasher callables.
  //!
  //! @param __hash1 First hasher
  //! @param __hash2 Second hasher
  _CCCL_HOST_DEVICE_API constexpr double_hashing(const _Hash1& __hash1 = {}, const _Hash2& __hash2 = {1})
      : __hash1{__hash1}
      , __hash2{__hash2}
  {}

  //! @brief Constructs a double hashing probing scheme with the given hasher tuple.
  //!
  //! @param __hash Hasher tuple
  _CCCL_HOST_DEVICE_API constexpr double_hashing(const ::cuda::std::tuple<_Hash1, _Hash2>& __hash)
      : __hash1{::cuda::std::get<0>(__hash)}
      , __hash2{::cuda::std::get<1>(__hash)}
  {}

  //! @brief Makes a copy of the current probing scheme with the given hasher.
  //!
  //! @tparam _NewHash Tuple-like new hasher type
  //!
  //! @param __hash Hasher
  //!
  //! @return Copy of the current probing scheme
  template <class _NewHash,
            class _Enable = ::cuda::std::enable_if_t<::cuda::experimental::cuco::is_tuple_like<_NewHash>::value>>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto rebind_hash_function(const _NewHash& __hash) const
  {
    static_assert(
      ::cuda::experimental::cuco::is_tuple_like<_NewHash>::value && ::cuda::std::tuple_size<_NewHash>::value == 2,
      "The given hasher must be a tuple-like object with exactly two elements");

    auto const& [__hash1, __hash2] = __hash;
    using __hash1_type             = ::cuda::std::decay_t<decltype(__hash1)>;
    using __hash2_type             = ::cuda::std::decay_t<decltype(__hash2)>;
    return double_hashing<cg_size, __hash1_type, __hash2_type>{__hash1, __hash2};
  }

  //! @brief Returns a probing iterator.
  //!
  //! @tparam _BucketSize Size of the bucket
  //! @tparam _ProbeKey Type of probing key
  //! @tparam _Extent Type of extent
  //!
  //! @param __probe_key The probing key
  //! @param __upper_bound Upper bound of the iteration
  //!
  //! @return An iterator whose value_type is convertible to the slot index type
  template <int _BucketSize, class _ProbeKey, class _Extent>
  _CCCL_HOST_DEVICE_API constexpr auto make_iterator(_ProbeKey __probe_key, _Extent __upper_bound) const noexcept
  {
    using __size_type = typename _Extent::index_type;
    return ::cuda::experimental::cuco::__detail::__probing_iterator<_Extent>{
      static_cast<__size_type>(__hash1(__probe_key)) % (__upper_bound.extent(0) / _BucketSize) * _BucketSize,
      static_cast<__size_type>((__hash2(__probe_key) % (__upper_bound.extent(0) / _BucketSize - 1) + 1) * _BucketSize),
      __upper_bound};
  }

  //! @brief Returns a cooperative group based probing iterator.
  //!
  //! @tparam _BucketSize Size of the bucket
  //! @tparam _ProbeKey Type of probing key
  //! @tparam _Extent Type of extent
  //! @tparam _ParentCG Type of parent cooperative group
  //!
  //! @param __group The cooperative group used to generate the probing iterator
  //! @param __probe_key The probing key
  //! @param __upper_bound Upper bound of the iteration
  //!
  //! @return An iterator whose value_type is convertible to the slot index type
  template <int _BucketSize, class _ProbeKey, class _Extent, class _ParentCG>
  _CCCL_HOST_DEVICE_API constexpr auto
  make_iterator(::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group,
                _ProbeKey __probe_key,
                _Extent __upper_bound) const noexcept
  {
    using __size_type              = typename _Extent::index_type;
    constexpr __size_type __stride = cg_size * _BucketSize;

    return ::cuda::experimental::cuco::__detail::__probing_iterator<_Extent>{
      static_cast<__size_type>(__hash1(__probe_key)) % (__upper_bound.extent(0) / __stride) * __stride
        + static_cast<__size_type>(__group.thread_rank() * _BucketSize),
      static_cast<__size_type>((__hash2(__probe_key) % (__upper_bound.extent(0) / __stride - 1) + 1) * __stride),
      __upper_bound};
  }

  //! @brief Gets the functions used to hash keys.
  //!
  //! @return The functions used to hash keys
  _CCCL_HOST_DEVICE_API constexpr hasher hash_function() const noexcept
  {
    return {__hash1, __hash2};
  }

private:
  _Hash1 __hash1;
  _Hash2 __hash2;
};

//! @brief Trait indicating whether a probing scheme is double hashing.
//!
//! @tparam _Tp Input probing scheme type
template <class _Tp>
struct is_double_hashing : ::cuda::std::false_type
{};

//! @brief Trait indicating whether a probing scheme is double hashing.
//!
//! @tparam _CgSize Cooperative group size
//! @tparam _Hash1 First hash functor
//! @tparam _Hash2 Second hash functor
template <int _CgSize, class _Hash1, class _Hash2>
struct is_double_hashing<double_hashing<_CgSize, _Hash1, _Hash2>> : ::cuda::std::true_type
{};
} // namespace cuda::experimental::cuco

#endif // _CUDAX___CUCO_PROBING_SCHEME_CUH
