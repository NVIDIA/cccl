//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__CUCO_HYPERLOGLOG_REF_CUH
#define _CUDAX__CUCO_HYPERLOGLOG_REF_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>
#include <cuda/stream>

#include <cuda/experimental/__cuco/detail/hyperloglog/hyperloglog_impl.cuh>
#include <cuda/experimental/__cuco/hash_functions.cuh>

#include <cooperative_groups.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! A strong type wrapper `cuda::experimental::cuco::sketch_size_kb` of `double`, for specifying the upper-bound sketch
//! size of `cuda::experimental::cuco::hyperloglog(_ref)` in KB.
//!
//! Note: Values can also be specified as literals, e.g., 64.3_KB.
using sketch_size_kb = detail::sketch_size_kb;

//! A strong type wrapper `cuda::experimental::cuco::standard_deviation` of `double`, for specifying the desired
//! standard deviation for the cardinality estimate of `cuda::experimental::cuco::hyperloglog(_ref)`.
using standard_deviation = detail::standard_deviation;

//! @brief A non-owning reference to a HyperLogLog sketch for approximating the number of distinct
//! items in a multiset.
//!
//! @note This implementation is based on the HyperLogLog++ algorithm:
//! https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf.
//!
//! @tparam _T Type of items to count
//! @tparam _Scope The scope in which operations will be performed by individual threads
//! @tparam _Hash Hash function used to hash items
template <class _T,
          ::cuda::thread_scope _Scope = ::cuda::thread_scope_device,
          class _Hash = ::cuda::experimental::cuco::hash<_T, ::cuda::experimental::cuco::hash_algorithm::xxhash_64>>
class hyperloglog_ref
{
  using __impl_type = detail::_HyperLogLog_Impl<_T, _Scope, _Hash>;

public:
  static constexpr auto thread_scope = __impl_type::thread_scope; ///< CUDA thread scope

  using value_type    = typename __impl_type::value_type; ///< Type of items to count
  using hasher        = typename __impl_type::hasher; ///< Type of hash function
  using register_type = typename __impl_type::register_type; ///< HLL register type

  template <::cuda::thread_scope _NewScope>
  using with_scope = hyperloglog_ref<_T, _NewScope, _Hash>; ///< Ref type with different thread scope

  //! @brief Constructs a non-owning `hyperloglog_ref` object.
  //!
  //! @throw If sketch size < 0.0625KB or 64B or standard deviation > 0.2765. Throws if called from
  //! host; UB if called from device.
  //! @throw If sketch storage has insufficient alignment. Throws if called from host; UB if called
  //! from device.
  //!
  //! @param __sketch_span Reference to sketch storage
  //! @param __hash The hash function used to hash items
  _CCCL_API constexpr hyperloglog_ref(::cuda::std::span<::cuda::std::byte> __sketch_span, _Hash const& __hash = {})
      : __impl{__sketch_span, __hash}
  {}

  //! @brief Resets the estimator, i.e., clears the current count estimate.
  //!
  //! @tparam _CG CUDA Cooperative Group type
  //!
  //! @param __group CUDA Cooperative group this operation is executed in
  template <class _CG>
  _CCCL_DEVICE constexpr void clear(_CG __group) noexcept
  {
    __impl.__clear(__group);
  }

  //! @brief Asynchronously resets the estimator, i.e., clears the current count estimate.
  //!
  //! @param __stream CUDA stream this operation is executed in
  _CCCL_HOST constexpr void clear_async(::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}}) noexcept
  {
    __impl.__clear_async(__stream);
  }

  //! @brief Resets the estimator, i.e., clears the current count estimate.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `clear_async`.
  //!
  //! @param __stream CUDA stream this operation is executed in
  _CCCL_HOST constexpr void clear(::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
  {
    __impl.__clear(__stream);
  }

  //! @brief Adds an item to the estimator.
  //!
  //! @param __item The item to be counted
  _CCCL_DEVICE constexpr void add(_T const& __item) noexcept
  {
    __impl.__add(__item);
  }

  //! @brief Asynchronously adds to be counted items to the estimator.
  //!
  //! @tparam _InputIt Device accessible random access input iterator where
  //! <tt>std::is_convertible<std::iterator_traits<_InputIt>::value_type,
  //! _T></tt> is `true`
  //!
  //! @param __first Beginning of the sequence of items
  //! @param __last End of the sequence of items
  //! @param __stream CUDA stream this operation is executed in
  template <class _InputIt>
  _CCCL_HOST constexpr void
  add_async(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
  {
    __impl.__add_async(__first, __last, __stream);
  }

  //! @brief Adds to be counted items to the estimator.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `add_async`.
  //!
  //! @tparam _InputIt Device accessible random access input iterator where
  //! <tt>std::is_convertible<std::iterator_traits<_InputIt>::value_type,
  //! _T></tt> is `true`
  //!
  //! @param __first Beginning of the sequence of items
  //! @param __last End of the sequence of items
  //! @param __stream CUDA stream this operation is executed in
  template <class _InputIt>
  _CCCL_HOST constexpr void
  add(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
  {
    __impl.__add(__first, __last, __stream);
  }

  //! @brief Merges the result of `other` estimator reference into `*this` estimator reference.
  //!
  //! @throw If this->sketch_bytes() != other.sketch_bytes() then behavior is undefined
  //!
  //! @tparam _CG CUDA Cooperative Group type
  //! @tparam _OtherScope Thread scope of `other` estimator
  //!
  //! @param __group CUDA Cooperative group this operation is executed in
  //! @param __other Other estimator reference to be merged into `*this`
  template <class _CG, ::cuda::thread_scope _OtherScope>
  _CCCL_DEVICE constexpr void merge(_CG __group, hyperloglog_ref<_T, _OtherScope, _Hash> const& __other)
  {
    __impl.__merge(__group, __other.__impl);
  }

  //! @brief Compute the estimated distinct items count.
  //!
  //! @param __group CUDA thread block group this operation is executed in
  //!
  //! @return Approximate distinct items count
  [[nodiscard]] _CCCL_DEVICE std::size_t estimate(cooperative_groups::thread_block const& __group) const noexcept
  {
    return __impl.__estimate(__group);
  }

  //! @brief Compute the estimated distinct items count.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @param __stream CUDA stream this operation is executed in
  //!
  //! @return Approximate distinct items count
  [[nodiscard]] _CCCL_HOST constexpr std::size_t
  estimate(::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}}) const
  {
    return __impl.__estimate(__stream);
  }

  //! @brief Gets the hash function.
  //!
  //! @return The hash function
  [[nodiscard]] _CCCL_API constexpr auto hash_function() const noexcept
  {
    return __impl.__hash_function();
  }

  //! @brief Gets the span of the sketch.
  //!
  //! @return The ::cuda::std::span of the sketch
  [[nodiscard]] _CCCL_API constexpr ::cuda::std::span<::cuda::std::byte> sketch() const noexcept
  {
    return __impl.__sketch_span();
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] _CCCL_API constexpr std::size_t sketch_bytes() const noexcept
  {
    return __impl.__sketch_bytes();
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @param __sketch_size_kb Upper bound sketch size in KB
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] _CCCL_API static constexpr std::size_t
  sketch_bytes(::cuda::experimental::cuco::sketch_size_kb __sketch_size_kb) noexcept
  {
    return __impl_type::__sketch_bytes(__sketch_size_kb);
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @param __standard_deviation Upper bound standard deviation for approximation error
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] _CCCL_API static constexpr std::size_t
  sketch_bytes(::cuda::experimental::cuco::standard_deviation __standard_deviation) noexcept
  {
    return __impl_type::sketch_bytes(__standard_deviation);
  }

  //! @brief Gets the alignment required for the sketch storage.
  //!
  //! @return The required alignment
  [[nodiscard]] _CCCL_API static constexpr std::size_t sketch_alignment() noexcept
  {
    return __impl_type::__sketch_alignment();
  }

private:
  __impl_type __impl; ///< Implementation object

  template <class _T_, ::cuda::thread_scope _Scope_, class _Hash_>
  friend class hyperloglog_ref;
};
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_HYPERLOGLOG_REF_CUH
