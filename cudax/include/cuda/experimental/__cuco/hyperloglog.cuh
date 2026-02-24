//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_HYPERLOGLOG_CUH
#define _CUDAX___CUCO_HYPERLOGLOG_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__stream/stream_ref.h>
#include <cuda/__utility/in_range.h>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__cccl/assert.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__utility/forward.h>

#include <cuda/experimental/__cuco/hash_functions.cuh>
#include <cuda/experimental/__cuco/hyperloglog_ref.cuh>
#include <cuda/experimental/container.cuh>
#include <cuda/experimental/memory_resource.cuh>
#include <cuda/experimental/stream.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! @brief A GPU-accelerated utility for approximating the number of distinct items in a multiset.
//!
//! @note This implementation is based on the HyperLogLog++ algorithm:
//! https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf.
//!
//! @tparam _Tp Type of items to count
//! @tparam _MemoryResource Type of memory resource used for device storage
//! @tparam _Scope The scope in which operations will be performed by individual threads
//! @tparam _Hash Hash function used to hash items
template <class _Tp,
          class _MemoryResource       = ::cuda::device_memory_pool_ref,
          ::cuda::thread_scope _Scope = ::cuda::thread_scope_device,
          class _Hash = ::cuda::experimental::cuco::hash<_Tp, ::cuda::experimental::cuco::hash_algorithm::xxhash_64>>
class hyperloglog
{
public:
  static constexpr auto thread_scope = _Scope; ///< CUDA thread scope

  template <::cuda::thread_scope _NewScope = thread_scope>
  using ref_type = hyperloglog_ref<_Tp, _NewScope, _Hash>; ///< Non-owning reference type

  using value_type    = typename ref_type<>::value_type; ///< Type of items to count
  using hasher        = typename ref_type<>::hasher; ///< Hash function type
  using register_type = typename ref_type<>::register_type; ///< HLL register type

  //! A strong type wrapper `sketch_size_kb` of `double`, for specifying the upper-bound
  //! sketch size of `cuda::experimental::cuco::hyperloglog(_ref)` in KB.
  //!
  //! @note Valid sketch sizes are in [0.0625 KB, 1024 KB], which correspond to precision [4, 18].
  using sketch_size_kb = ::cuda::experimental::cuco::__sketch_size_kb_t;

  //! A strong type wrapper `standard_deviation` of `double`, for specifying the desired
  //! standard deviation for the cardinality estimate of `cuda::experimental::cuco::hyperloglog(_ref)`.
  //!
  //! @note Valid standard deviations are approximately in [0.00216, 0.2765], which correspond to
  //! precision [4, 18].
  using standard_deviation = ::cuda::experimental::cuco::__standard_deviation_t;

  //! A strong type wrapper `precision` of `int`, for specifying the HyperLogLog precision
  //! parameter of `cuda::experimental::cuco::hyperloglog(_ref)`.
  //!
  //! @note Valid precision values are in [4, 18], which correspond to sketch sizes in
  //! [0.0625 KB, 1024 KB] and standard deviations approximately in [0.00216, 0.2765].
  using precision = ::cuda::experimental::cuco::__precision_t;

private:
  ::cuda::device_buffer<register_type> __sketch_buffer; ///< Storage for sketch
  ref_type<> __ref; ///< Device ref of the current `hyperloglog` object

  // Needs to be friends with other instantiations of this class template to have access to their
  // storage
  template <class _Tp_, class _MemoryResource_, ::cuda::thread_scope _Scope_, class _Hash_>
  friend class hyperloglog;

public:
  // TODO enable CTAD
  //! @brief Constructs a `hyperloglog` host object.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @param __memory_resource A memory resource used for allocating device storage
  //! @param __sketch_size_kb Maximum sketch size in KB
  //! @param __hash The hash function used to hash items
  //! @param __stream CUDA stream used to initialize the object
  //!
  //! @throw If sketch size implies precision outside [4, 18].
  template <typename _MemoryResource_ = _MemoryResource>
  constexpr hyperloglog(_MemoryResource_&& __memory_resource,
                        sketch_size_kb __sketch_size_kb = sketch_size_kb{32.0},
                        const _Hash& __hash             = {},
                        ::cuda::stream_ref __stream     = ::cuda::stream_ref{cudaStream_t{nullptr}})
      : hyperloglog{
          ::cuda::std::forward<_MemoryResource_>(__memory_resource), __to_precision(__sketch_size_kb), __hash, __stream}
  {}

  //! @brief Constructs a `hyperloglog` host object.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @param __sketch_size_kb Maximum sketch size in KB
  //! @param __hash The hash function used to hash items
  //! @param __stream CUDA stream used to initialize the object
  //!
  //! @throw If sketch size implies precision outside [4, 18].
  constexpr hyperloglog(sketch_size_kb __sketch_size_kb = sketch_size_kb{32.0},
                        const _Hash& __hash             = {},
                        ::cuda::stream_ref __stream     = ::cuda::stream_ref{cudaStream_t{nullptr}})
      : hyperloglog{__to_precision(__sketch_size_kb), __hash, __stream}
  {}

  //! @brief Constructs a `hyperloglog` host object.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @param __memory_resource A memory resource used for allocating device storage
  //! @param __sd Desired standard deviation for the approximation error
  //! @param __hash The hash function used to hash items
  //! @param __stream CUDA stream used to initialize the object
  //!
  //! @throw If standard deviation implies precision outside [4, 18].
  template <typename _MemoryResource_ = _MemoryResource>
  constexpr hyperloglog(_MemoryResource_&& __memory_resource,
                        standard_deviation __sd,
                        const _Hash& __hash         = {},
                        ::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
      : hyperloglog{::cuda::std::forward<_MemoryResource_>(__memory_resource), __to_precision(__sd), __hash, __stream}
  {}

  //! @brief Constructs a `hyperloglog` host object.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @param __memory_resource A memory resource used for allocating device storage
  //! @param __precision HyperLogLog precision parameter (determines number of registers as 2^precision)
  //! @param __hash The hash function used to hash items
  //! @param __stream CUDA stream used to initialize the object
  //!
  //! @throw If precision is outside [4, 18].
  template <typename _MemoryResource_ = _MemoryResource>
  constexpr hyperloglog(_MemoryResource_&& __memory_resource,
                        precision __precision,
                        const _Hash& __hash         = {},
                        ::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
      : __sketch_buffer{__stream,
                        ::cuda::std::forward<_MemoryResource_>(__memory_resource),
                        ref_type<>::sketch_bytes(
                          __precision_in_bounds(__precision, "HyperLogLog precision must be in [4, 18]"))
                          / sizeof(register_type),
                        ::cuda::no_init}
      , __ref{::cuda::std::as_writable_bytes(::cuda::std::span{__sketch_buffer.data(), __sketch_buffer.size()}), __hash}
  {
    clear_async(__stream);
  }

  //! @brief Constructs a `hyperloglog` host object.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @param __sd Desired standard deviation for the approximation error
  //! @param __hash The hash function used to hash items
  //! @param __stream CUDA stream used to initialize the object
  //!
  //! @throw If standard deviation implies precision outside [4, 18].
  constexpr hyperloglog(standard_deviation __sd,
                        const _Hash& __hash         = {},
                        ::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
      : hyperloglog{__to_precision(__sd), __hash, __stream}
  {}

  //! @brief Constructs a `hyperloglog` host object.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @param __precision HyperLogLog precision parameter (determines number of registers as 2^precision)
  //! @param __hash The hash function used to hash items
  //! @param __stream CUDA stream used to initialize the object
  //!
  //! @throw If precision is outside [4, 18].
  constexpr hyperloglog(precision __precision,
                        const _Hash& __hash         = {},
                        ::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
      : __sketch_buffer{__stream,
                        ::cuda::device_default_memory_pool(::cuda::device_ref{0}),
                        ref_type<>::sketch_bytes(
                          __precision_in_bounds(__precision, "HyperLogLog precision must be in [4, 18]"))
                          / sizeof(register_type),
                        ::cuda::no_init}
      , __ref{::cuda::std::as_writable_bytes(::cuda::std::span{__sketch_buffer.data(), __sketch_buffer.size()}), __hash}
  {
    clear_async(__stream);
  }

  ~hyperloglog() = default;

  hyperloglog(const hyperloglog&) = delete;
  //! @brief Copy-assignment operator.
  //!
  //! @return Copy of `*this`
  hyperloglog& operator=(const hyperloglog&) = delete;
  hyperloglog(hyperloglog&&)                 = default; ///< Move constructor

  hyperloglog& operator=(hyperloglog&&) = default;

  //! @brief Asynchronously resets the estimator, i.e., clears the current count estimate.
  //!
  //! @param __stream CUDA stream this operation is executed in
  constexpr void clear_async(::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}}) noexcept
  {
    __ref.clear_async(__stream);
  }

  //! @brief Resets the estimator, i.e., clears the current count estimate.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `clear_async`.
  //!
  //! @param __stream CUDA stream this operation is executed in
  constexpr void clear(::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
  {
    __ref.clear(__stream);
  }

  //! @brief Asynchronously adds to be counted items to the estimator.
  //!
  //! @tparam _InputIt Device accessible random access input iterator where
  //! <tt>std::is_convertible<std::iterator_traits<_InputIt>::value_type,
  //! _Tp></tt> is `true`
  //!
  //! @param __first Beginning of the sequence of items
  //! @param __last End of the sequence of items
  //! @param __stream CUDA stream this operation is executed in
  template <class _InputIt>
  constexpr void
  add_async(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
  {
    __ref.add_async(__first, __last, __stream);
  }

  //! @brief Adds to be counted items to the estimator.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `add_async`.
  //!
  //! @tparam _InputIt Device accessible random access input iterator where
  //! <tt>std::is_convertible<std::iterator_traits<_InputIt>::value_type,
  //! _Tp></tt> is `true`
  //!
  //! @param __first Beginning of the sequence of items
  //! @param __last End of the sequence of items
  //! @param __stream CUDA stream this operation is executed in
  template <class _InputIt>
  constexpr void
  add(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
  {
    __ref.add(__first, __last, __stream);
  }

  //! @brief Asynchronously merges the result of `other` estimator into `*this` estimator.
  //!
  //! @throw If sketch_bytes() != __other.sketch_bytes()
  //!
  //! @tparam _OtherScope Thread scope of `other` estimator
  //! @tparam _OtherMemoryResource Memory resource type of `other` estimator
  //!
  //! @param __other Other estimator to be merged into `*this`
  //! @param __stream CUDA stream this operation is executed in
  template <::cuda::thread_scope _OtherScope, class _OtherMemoryResource>
  constexpr void merge_async(const hyperloglog<_Tp, _OtherMemoryResource, _OtherScope, _Hash>& __other,
                             ::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
  {
    __ref.merge_async(__other.__ref, __stream);
  }

  //! @brief Merges the result of `other` estimator into `*this` estimator.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `merge_async`.
  //!
  //! @throw If sketch_bytes() != __other.sketch_bytes()
  //!
  //! @tparam _OtherScope Thread scope of `other` estimator
  //! @tparam _OtherMemoryResource Memory resource type of `other` estimator
  //!
  //! @param __other Other estimator to be merged into `*this`
  //! @param __stream CUDA stream this operation is executed in
  template <::cuda::thread_scope _OtherScope, class _OtherMemoryResource>
  constexpr void merge(const hyperloglog<_Tp, _OtherMemoryResource, _OtherScope, _Hash>& __other,
                       ::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
  {
    __ref.merge(__other.__ref, __stream);
  }

  //! @brief Asynchronously merges the result of `other` estimator reference into `*this` estimator.
  //!
  //! @throw If sketch_bytes() != __other.sketch_bytes()
  //!
  //! @tparam _OtherScope Thread scope of `other` estimator
  //!
  //! @param __other_ref Other estimator reference to be merged into `*this`
  //! @param __stream CUDA stream this operation is executed in
  template <::cuda::thread_scope _OtherScope>
  constexpr void merge_async(const ref_type<_OtherScope>& __other_ref,
                             ::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
  {
    __ref.merge_async(__other_ref, __stream);
  }

  //! @brief Merges the result of `other` estimator reference into `*this` estimator.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `merge_async`.
  //!
  //! @throw If sketch_bytes() != __other.sketch_bytes()
  //!
  //! @tparam _OtherScope Thread scope of `other` estimator
  //!
  //! @param __other_ref Other estimator reference to be merged into `*this`
  //! @param __stream CUDA stream this operation is executed in
  template <::cuda::thread_scope _OtherScope>
  constexpr void merge(const ref_type<_OtherScope>& __other_ref,
                       ::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
  {
    __ref.merge(__other_ref, __stream);
  }

  //! @brief Compute the estimated distinct items count.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @tparam _MemoryResource Host memory resource used for allocating the host buffer required to
  //! compute the final estimate by copying the sketch from device to host
  //!
  //! @param __host_mr Host memory resource used for copying the sketch
  //! @param __stream CUDA stream this operation is executed in
  //!
  //! @return Approximate distinct items count
  template <typename _HostMemoryResource = ::cuda::mr::legacy_pinned_memory_resource>
  [[nodiscard]] constexpr ::cuda::std::size_t estimate(
    _HostMemoryResource __host_mr = {}, ::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}}) const
  {
    return __ref.estimate(__host_mr, __stream);
  }

  //! @brief Get device ref.
  //!
  //! @return Device ref object of the current `hyperloglog` host object
  [[nodiscard]] constexpr ref_type<> ref() const noexcept
  {
    return {sketch(), hash_function()};
  }

  //! @brief Get hash function.
  //!
  //! @return The hash function
  [[nodiscard]] constexpr auto hash_function() const noexcept
  {
    return __ref.hash_function();
  }

  //! @brief Gets the span of the sketch.
  //!
  //! @return The ::cuda::std::span of the sketch
  [[nodiscard]] constexpr ::cuda::std::span<::cuda::std::byte> sketch() const noexcept
  {
    return __ref.sketch();
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] constexpr ::cuda::std::size_t sketch_bytes() const noexcept
  {
    return __ref.sketch_bytes();
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @param __sketch_size_kb Upper bound sketch size in KB
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] static constexpr ::cuda::std::size_t sketch_bytes(sketch_size_kb __sketch_size_kb) noexcept
  {
    return ref_type<>::sketch_bytes(__sketch_size_kb);
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @param __standard_deviation Upper bound standard deviation for approximation error
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] static constexpr ::cuda::std::size_t sketch_bytes(standard_deviation __standard_deviation) noexcept
  {
    return ref_type<>::sketch_bytes(__standard_deviation);
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @param __precision HyperLogLog precision parameter
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] static constexpr ::cuda::std::size_t sketch_bytes(precision __precision) noexcept
  {
    return ref_type<>::sketch_bytes(__precision);
  }

  //! @brief Gets the alignment required for the sketch storage.
  //!
  //! @return The required alignment
  [[nodiscard]] static constexpr ::cuda::std::size_t sketch_alignment() noexcept
  {
    return ref_type<>::sketch_alignment();
  }

private:
  [[nodiscard]] static constexpr precision __precision_in_bounds(precision __precision, const char* __message)
  {
    const auto __value    = static_cast<int>(__precision);
    const auto __in_range = ::cuda::in_range(__value, 4, 18);
    if (!__in_range)
    {
      _CCCL_THROW(::std::invalid_argument, __message);
    }
    return __precision;
  }

  [[nodiscard]] static constexpr precision __to_precision(sketch_size_kb __sketch_size_kb)
  {
    const auto __bytes     = ref_type<>::sketch_bytes(__sketch_size_kb) / sizeof(register_type);
    const auto __precision = static_cast<int>(::cuda::std::countr_zero(static_cast<::cuda::std::size_t>(__bytes)));
    return __precision_in_bounds(
      precision{__precision}, "HyperLogLog sketch size must be in range [0.0625 KB, 1024 KB]");
  }

  [[nodiscard]] static constexpr precision __to_precision(standard_deviation __standard_deviation)
  {
    const auto __bytes     = ref_type<>::sketch_bytes(__standard_deviation) / sizeof(register_type);
    const auto __precision = static_cast<int>(::cuda::std::countr_zero(static_cast<::cuda::std::size_t>(__bytes)));
    return __precision_in_bounds(
      precision{__precision}, "HyperLogLog standard deviation must be in range [0.00216, 0.2765]");
  }
};
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_HYPERLOGLOG_CUH
