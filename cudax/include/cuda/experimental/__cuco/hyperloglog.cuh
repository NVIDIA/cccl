//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__CUCO_HYPERLOGLOG_CUH
#define _CUDAX__CUCO_HYPERLOGLOG_CUH

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

#include <cuda/experimental/__cuco/hash_functions.cuh>
#include <cuda/experimental/__cuco/hyperloglog_ref.cuh>
#include <cuda/experimental/container.cuh>
#include <cuda/experimental/memory_resource.cuh>
#include <cuda/experimental/stream.cuh>

// TODO : remove these includes
#include <iterator>
#include <memory>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! @brief A GPU-accelerated utility for approximating the number of distinct items in a multiset.
//!
//! @note This implementation is based on the HyperLogLog++ algorithm:
//! https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf.
//!
//! @tparam _Tp Type of items to count
//! @tparam _MemoryResourceRef Type of non-owning memory resource used for device storage
//! @tparam _Scope The scope in which operations will be performed by individual threads
//! @tparam _Hash Hash function used to hash items
template <class _Tp,
          class _MemoryResourceRef    = ::cuda::device_memory_pool_ref,
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
  using sketch_size_kb = ::cuda::experimental::cuco::__sketch_size_kb_t;

  //! A strong type wrapper `standard_deviation` of `double`, for specifying the desired
  //! standard deviation for the cardinality estimate of `cuda::experimental::cuco::hyperloglog(_ref)`.
  using standard_deviation = ::cuda::experimental::cuco::__standard_deviation_t;

  // TODO enable CTAD
  //! @brief Constructs a `hyperloglog` host object.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @param __memory_resource_ref A non-owning memory resource used for allocating device storage
  //! @param __sketch_size_kb Maximum sketch size in KB
  //! @param __hash The hash function used to hash items
  //! @param __stream CUDA stream used to initialize the object
  constexpr hyperloglog(_MemoryResourceRef __memory_resource_ref,
                        sketch_size_kb __sketch_size_kb = sketch_size_kb{32.0},
                        const _Hash& __hash             = {},
                        ::cuda::stream_ref __stream     = ::cuda::stream_ref{cudaStream_t{nullptr}})
      : __memory_resource_ref(__memory_resource_ref)
      , __sketch_buffer{__stream,
                        __memory_resource_ref,
                        ref_type<>::sketch_bytes(__sketch_size_kb) / sizeof(register_type),
                        ::cuda::experimental::no_init}
      , __ref{::cuda::std::span{reinterpret_cast<::cuda::std::byte*>(__sketch_buffer.data()),
                                ref_type<>::sketch_bytes(__sketch_size_kb)},
              __hash}
  {
    this->clear_async(__stream);
  }

  //! @brief Constructs a `hyperloglog` host object.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @param __sketch_size_kb Maximum sketch size in KB
  //! @param __hash The hash function used to hash items
  //! @param __stream CUDA stream used to initialize the object
  constexpr hyperloglog(sketch_size_kb __sketch_size_kb = sketch_size_kb{32.0},
                        const _Hash& __hash             = {},
                        ::cuda::stream_ref __stream     = ::cuda::stream_ref{cudaStream_t{nullptr}})
      : __memory_resource_ref(::cuda::device_default_memory_pool(::cuda::device_ref{0}))
      , __sketch_buffer{__stream,
                        __memory_resource_ref,
                        ref_type<>::sketch_bytes(__sketch_size_kb) / sizeof(register_type),
                        ::cuda::experimental::no_init}
      , __ref{::cuda::std::span{reinterpret_cast<::cuda::std::byte*>(__sketch_buffer.data()),
                                ref_type<>::sketch_bytes(__sketch_size_kb)},
              __hash}
  {
    this->clear_async(__stream);
  }

  //! @brief Constructs a `hyperloglog` host object.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @param __memory_resource_ref A non-owning memory resource used for allocating device storage
  //! @param __sd Desired standard deviation for the approximation error
  //! @param __hash The hash function used to hash items
  //! @param __stream CUDA stream used to initialize the object
  constexpr hyperloglog(_MemoryResourceRef __memory_resource_ref,
                        standard_deviation __sd,
                        const _Hash& __hash         = {},
                        ::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
      : __memory_resource_ref(__memory_resource_ref)
      , __sketch_buffer{__stream,
                        __memory_resource_ref,
                        ref_type<>::sketch_bytes(__sd) / sizeof(register_type),
                        ::cuda::experimental::no_init}
      , __ref{::cuda::std::span{reinterpret_cast<::cuda::std::byte*>(__sketch_buffer.data()),
                                ref_type<>::sketch_bytes(__sd)},
              __hash}
  {
    this->clear_async(__stream);
  }

  //! @brief Constructs a `hyperloglog` host object.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @param __sd Desired standard deviation for the approximation error
  //! @param __hash The hash function used to hash items
  //! @param __stream CUDA stream used to initialize the object
  constexpr hyperloglog(standard_deviation __sd,
                        const _Hash& __hash         = {},
                        ::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}})
      : __memory_resource_ref(::cuda::device_default_memory_pool(::cuda::device_ref{0}))
      , __sketch_buffer{__stream,
                        __memory_resource_ref,
                        ref_type<>::sketch_bytes(__sd) / sizeof(register_type),
                        ::cuda::experimental::no_init}
      , __ref{::cuda::std::span{reinterpret_cast<::cuda::std::byte*>(__sketch_buffer.data()),
                                ref_type<>::sketch_bytes(__sd)},
              __hash}
  {
    this->clear_async(__stream);
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

  //! @brief Compute the estimated distinct items count.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @param __stream CUDA stream this operation is executed in
  //!
  //! @return Approximate distinct items count
  [[nodiscard]] constexpr std::size_t
  estimate(::cuda::stream_ref __stream = ::cuda::stream_ref{cudaStream_t{nullptr}}) const
  {
    return __ref.estimate(__stream);
  }

  //! @brief Get device ref.
  //!
  //! @return Device ref object of the current `hyperloglog` host object
  [[nodiscard]] constexpr ref_type<> ref() const noexcept
  {
    return {this->sketch(), this->hash_function()};
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
  [[nodiscard]] constexpr std::size_t sketch_bytes() const noexcept
  {
    return __ref.sketch_bytes();
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @param __sketch_size_kb Upper bound sketch size in KB
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] static constexpr std::size_t sketch_bytes(sketch_size_kb __sketch_size_kb) noexcept
  {
    return ref_type<>::sketch_bytes(__sketch_size_kb);
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @param __standard_deviation Upper bound standard deviation for approximation error
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] static constexpr std::size_t sketch_bytes(standard_deviation __standard_deviation) noexcept
  {
    return ref_type<>::sketch_bytes(__standard_deviation);
  }

  //! @brief Gets the alignment required for the sketch storage.
  //!
  //! @return The required alignment
  [[nodiscard]] static constexpr std::size_t sketch_alignment() noexcept
  {
    return ref_type<>::sketch_alignment();
  }

private:
  _MemoryResourceRef __memory_resource_ref; ///< Memory resource used to allocate device-accessible storage
  ::cuda::experimental::device_buffer<register_type> __sketch_buffer; ///< Storage for sketch
  ref_type<> __ref; ///< Device ref of the current `hyperloglog` object

  // Needs to be friends with other instantiations of this class template to have access to their
  // storage
  template <class _Tp_, class _MemoryResourceRef_, ::cuda::thread_scope _Scope_, class _Hash_>
  friend class hyperloglog;
};
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_HYPERLOGLOG_CUH
