//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___HYPERLOGLOG_IMPL_CUH
#define _CUDAX___CUCO___HYPERLOGLOG_IMPL_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__container/buffer.h>
#include <cuda/__memory/is_aligned.h>
#include <cuda/__memory_resource/legacy_pinned_memory_resource.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/__utility/in_range.h>
#include <cuda/atomic>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/span>

#include <cuda/experimental/__cuco/__hyperloglog/finalizer.cuh>
#include <cuda/experimental/__cuco/__hyperloglog/kernels.cuh>
#include <cuda/experimental/__cuco/__utility/strong_type.cuh>
#include <cuda/experimental/__cuco/hash_functions.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <cooperative_groups.h>

#include <cooperative_groups/reduce.h>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
CUDAX_CUCO_DEFINE_STRONG_TYPE(__sketch_size_kb_t, double);

CUDAX_CUCO_DEFINE_STRONG_TYPE(__standard_deviation_t, double);

CUDAX_CUCO_DEFINE_STRONG_TYPE(__precision_t, int);

//! @brief A GPU-accelerated utility for approximating the number of distinct items in a multiset.
//!
//! @note This class implements the HyperLogLog/HyperLogLog++ algorithm:
//! https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf.
//!
//! @tparam _Tp Type of items to count
//! @tparam _Scope The scope in which operations will be performed by individual threads
//! @tparam _Hash Hash function used to hash items
template <class _Tp, ::cuda::thread_scope _Scope, class _Hash>
class __hyperloglog_impl
{
  using __fp_type         = double; ///< Floating point type used for reduction
  using __hash_value_type = decltype(::cuda::std::declval<_Hash>()(::cuda::std::declval<_Tp>())); ///< Hash value type

public:
  using __value_type    = _Tp; ///< Type of items to count
  using __hasher        = _Hash; ///< Hash function type
  using __register_type = int; ///< HLL register type

private:
  __hasher __hash; ///< Hash function used to hash items
  int __precision; ///< HLL precision parameter
  ::cuda::std::span<__register_type> __sketch; ///< HLL sketch storage

  template <class _Tp_, ::cuda::thread_scope _Scope_, class _Hash_>
  friend struct __hyperloglog_impl;

public:
  static constexpr auto __thread_scope = _Scope; ///< CUDA thread scope

  template <::cuda::thread_scope _NewScope>
  using __with_scope = __hyperloglog_impl<_Tp, _NewScope, _Hash>; ///< Ref type with different thread scope

  //! @brief Constructs a non-owning `__hyperloglog_impl` object.
  //!
  //! @throw If sketch size < 0.0625KB or 64B or standard deviation > 0.2765. Throws if called from
  //! host; __trap() if called from device.
  //! @throw If sketch size implies precision outside [4, 18]. Throws if called from host; __trap() if
  //! called from device.
  //! @throw If sketch storage has insufficient alignment. Throws if called from host; __trap() if called from device.
  //!
  //! @param __sketch_span Reference to sketch storage
  //! @param __hash The hash function used to hash items
  _CCCL_API constexpr __hyperloglog_impl(::cuda::std::span<::cuda::std::byte> __sketch_span, const _Hash& __hash)
      : __hash{__hash}
      , __precision{::cuda::std::countr_zero(
          __sketch_bytes(static_cast<::cuda::experimental::cuco::__sketch_size_kb_t>(__sketch_span.size() / 1024.0))
          / sizeof(__register_type))}
      , __sketch{reinterpret_cast<int*>(__sketch_span.data()), __sketch_bytes() / sizeof(__register_type)}
  // MSVC fails with __register_type*, use int* instead
  {
    if (!::cuda::is_aligned(__sketch_span.data(), __sketch_alignment()))
    {
      _CCCL_THROW(::std::invalid_argument, "Sketch storage has insufficient alignment");
    }

    if (!::cuda::in_range(__precision, 4, 18))
    {
      _CCCL_THROW(::std::invalid_argument, "Minimum required sketch size is 0.0625KB or 64B");
    }
  }

  //! @brief Resets the estimator, i.e., clears the current count estimate.
  //!
  //! @tparam _CG CUDA Cooperative Group type
  //!
  //! @param __group CUDA Cooperative group this operation is executed in
  template <class _CG>
  _CCCL_DEVICE constexpr void __clear(_CG __group) noexcept
  {
    for (int __i = __group.thread_rank(); __i < __sketch.size(); __i += __group.size())
    {
      __sketch[__i] = 0;
    }
  }

  //! @brief Resets the estimator, i.e., clears the current count estimate.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `__clear_async`.
  //!
  //! @param __stream CUDA stream this operation is executed in
  _CCCL_HOST constexpr void __clear(::cuda::stream_ref __stream)
  {
    __clear_async(__stream);
    __stream.sync();
  }

  //! @brief Asynchronously resets the estimator, i.e., clears the current count estimate.
  //!
  //! @param __stream CUDA stream this operation is executed in
  _CCCL_HOST constexpr void __clear_async(::cuda::stream_ref __stream)
  {
    constexpr auto __block_size = 1024;
    ::cuda::experimental::cuco::__hyperloglog_ns::__clear<<<1, __block_size, 0, __stream.get()>>>(*this);
  }

  //! @brief Adds an item to the estimator.
  //!
  //! @param __item The item to be counted
  _CCCL_DEVICE constexpr void __add(const _Tp& __item) noexcept
  {
    const auto __h      = __hash(__item);
    const auto __reg    = __h & __register_mask();
    const auto __zeroes = ::cuda::std::countl_zero(__h | __register_mask()) + 1;

    // reversed order (same one as Spark uses)
    // const auto __reg    = __h >> ((sizeof(__hash_value_type) * 8) - __precision);
    // const auto __zeroes = ::cuda::std::countl_zero(__h << __precision) + 1;

    __update_max(__reg, __zeroes);
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
  _CCCL_HOST constexpr void __add_async(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream)
  {
    const auto __num_items = ::cuda::std::distance(__first, __last);
    if (__num_items == 0)
    {
      return;
    }

    int __grid_size         = 0;
    int __block_size        = 0;
    const int __shmem_bytes = __sketch_bytes();
    const void* __kernel    = nullptr;

    // In case the input iterator represents a contiguous memory segment we can employ efficient
    // vectorized loads
    if constexpr (::cuda::std::contiguous_iterator<_InputIt>)
    {
      const auto __ptr                  = ::cuda::std::to_address(__first);
      constexpr auto __max_vector_bytes = 32;
      const auto __alignment =
        1u << ::cuda::std::countr_zero(reinterpret_cast<::cuda::std::uintptr_t>(__ptr) | __max_vector_bytes);
      const auto __vector_size = __alignment / sizeof(__value_type);

      switch (__vector_size)
      {
        using ::cuda::experimental::cuco::__hyperloglog_ns::__add_shmem_vectorized;
        case 2:
          __kernel = reinterpret_cast<const void*>(__add_shmem_vectorized<2, __hyperloglog_impl>);
          break;
        case 4:
          __kernel = reinterpret_cast<const void*>(__add_shmem_vectorized<4, __hyperloglog_impl>);
          break;
        case 8:
          __kernel = reinterpret_cast<const void*>(__add_shmem_vectorized<8, __hyperloglog_impl>);
          break;
        case 16:
          __kernel = reinterpret_cast<const void*>(__add_shmem_vectorized<16, __hyperloglog_impl>);
          break;
      };
    }

    if (__kernel != nullptr && __try_reserve_shmem(__kernel, __shmem_bytes))
    {
      if constexpr (::cuda::std::contiguous_iterator<_InputIt>)
      {
        // We make use of the occupancy calculator to get the minimum number of blocks which still
        // saturates the GPU. This reduces the shmem initialization overhead and atomic contention
        // on the final register array during the merge phase.
        _CCCL_TRY_CUDA_API(
          ::cudaOccupancyMaxPotentialBlockSize,
          "cudaOccupancyMaxPotentialBlockSize failed",
          &__grid_size,
          &__block_size,
          __kernel,
          __shmem_bytes);

        const auto __ptr      = ::cuda::std::addressof(__first[0]);
        void* __kernel_args[] = {const_cast<void*>(reinterpret_cast<const void*>(&__ptr)),
                                 const_cast<void*>(reinterpret_cast<const void*>(&__num_items)),
                                 reinterpret_cast<void*>(this)};
        _CCCL_TRY_CUDA_API(
          ::cudaLaunchKernel,
          "cudaLaunchKernel failed",
          __kernel,
          __grid_size,
          __block_size,
          __kernel_args,
          __shmem_bytes,
          __stream.get());
      }
    }
    else
    {
      __kernel = reinterpret_cast<const void*>(
        ::cuda::experimental::cuco::__hyperloglog_ns::__add_shmem<_InputIt, __hyperloglog_impl>);
      void* __kernel_args[] = {const_cast<void*>(reinterpret_cast<const void*>(&__first)),
                               const_cast<void*>(reinterpret_cast<const void*>(&__num_items)),
                               reinterpret_cast<void*>(this)};
      if (__try_reserve_shmem(__kernel, __shmem_bytes))
      {
        _CCCL_TRY_CUDA_API(
          ::cudaOccupancyMaxPotentialBlockSize,
          "cudaOccupancyMaxPotentialBlockSize failed",
          &__grid_size,
          &__block_size,
          __kernel,
          __shmem_bytes);

        _CCCL_TRY_CUDA_API(
          ::cudaLaunchKernel,
          "cudaLaunchKernel failed",
          __kernel,
          __grid_size,
          __block_size,
          __kernel_args,
          __shmem_bytes,
          __stream.get());
      }
      else
      {
        // Computes sketch directly in global memory. (Fallback path in case there is not enough
        // shared memory available)
        __kernel = reinterpret_cast<const void*>(
          ::cuda::experimental::cuco::__hyperloglog_ns::__add_gmem<_InputIt, __hyperloglog_impl>);

        _CCCL_TRY_CUDA_API(
          ::cudaOccupancyMaxPotentialBlockSize,
          "cudaOccupancyMaxPotentialBlockSize failed",
          &__grid_size,
          &__block_size,
          __kernel,
          0);

        _CCCL_TRY_CUDA_API(
          ::cudaLaunchKernel,
          "cudaLaunchKernel failed",
          __kernel,
          __grid_size,
          __block_size,
          __kernel_args,
          0,
          __stream.get());
      }
    }
  }

  //! @brief Adds to be counted items to the estimator.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `__add_async`.
  //!
  //! @tparam _InputIt Device accessible random access input iterator where
  //! <tt>std::is_convertible<std::iterator_traits<_InputIt>::value_type,
  //! _Tp></tt> is `true`
  //!
  //! @param __first Beginning of the sequence of items
  //! @param __last End of the sequence of items
  //! @param __stream CUDA stream this operation is executed in
  template <class _InputIt>
  _CCCL_HOST constexpr void __add(_InputIt __first, _InputIt __last, ::cuda::stream_ref __stream)
  {
    __add_async(__first, __last, __stream);
    __stream.sync();
  }

  //! @brief Merges the result of `other` estimator reference into `*this` estimator reference.
  //!
  //! @throw If __sketch_bytes() != other.__sketch_bytes(), then terminates execution with a device __trap()
  //!
  //! @tparam _CG CUDA Cooperative Group type
  //! @tparam _OtherScope Thread scope of `other` estimator
  //!
  //! @param __group CUDA Cooperative group this operation is executed in
  //! @param __other Other estimator reference to be merged into `*this`
  template <class _CG, ::cuda::thread_scope _OtherScope>
  _CCCL_DEVICE constexpr void __merge(_CG __group, __hyperloglog_impl<_Tp, _OtherScope, _Hash>& __other)
  {
    if (__other.__precision != __precision)
    {
      _CCCL_THROW(::std::invalid_argument, "Cannot merge estimators with different sketch sizes");
    }

    for (int __i = __group.thread_rank(); __i < __sketch.size(); __i += __group.size())
    {
      __update_max(__i, __other.__sketch[__i]);
    }
  }

  //! @brief Asynchronously merges the result of `other` estimator reference into `*this`
  //! estimator.
  //!
  //! @throw If __sketch_bytes() != __other.__sketch_bytes(), then terminates execution with a device __trap()
  //!
  //! @tparam _OtherScope Thread scope of `other` estimator
  //!
  //! @param __other Other estimator reference to be merged into `*this`
  //! @param __stream CUDA stream this operation is executed in
  template <::cuda::thread_scope _OtherScope>
  _CCCL_HOST constexpr void
  __merge_async(const __hyperloglog_impl<_Tp, _OtherScope, _Hash>& __other, ::cuda::stream_ref __stream)
  {
    if (__other.__precision != __precision)
    {
      _CCCL_THROW(::std::invalid_argument, "Cannot merge estimators with different sketch sizes");
    }

    constexpr auto __block_size = 1024;
    ::cuda::experimental::cuco::__hyperloglog_ns::__merge<<<1, __block_size, 0, __stream.get()>>>(__other, *this);
  }

  //! @brief Merges the result of `other` estimator reference into `*this` estimator.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `__merge_async`.
  //!
  //! @throw If __sketch_bytes() != __other.__sketch_bytes(), then terminates execution with a device __trap()
  //!
  //! @tparam _OtherScope Thread scope of `other` estimator
  //!
  //! @param __other Other estimator reference to be merged into `*this`
  //! @param __stream CUDA stream this operation is executed in
  template <::cuda::thread_scope _OtherScope>
  _CCCL_HOST constexpr void
  __merge(const __hyperloglog_impl<_Tp, _OtherScope, _Hash>& __other, ::cuda::stream_ref __stream)
  {
    __merge_async(__other, __stream);
    __stream.sync();
  }

  //! @brief Compute the estimated distinct items count.
  //!
  //! @param __group CUDA thread block group this operation is executed in
  //!
  //! @return Approximate distinct items count
  [[nodiscard]] _CCCL_DEVICE ::cuda::std::size_t
  __estimate(const ::cooperative_groups::thread_block& __group) const noexcept
  {
    __shared__ ::cuda::atomic<__fp_type, ::cuda::std::thread_scope_block> __block_sum;
    __shared__ ::cuda::atomic<int, ::cuda::std::thread_scope_block> __block_zeroes;
    __shared__ ::cuda::std::size_t __estimate;

    if (__group.thread_rank() == 0)
    {
      __block_sum.store(0);
      __block_zeroes.store(0);
    }
    __group.sync();

    __fp_type __thread_sum = 0;
    int __thread_zeroes    = 0;
    for (int __i = __group.thread_rank(); __i < __sketch.size(); __i += __group.size())
    {
      const auto __reg = __sketch[__i];
      __thread_sum += __fp_type{1} / static_cast<__fp_type>(1u << __reg);
      __thread_zeroes += __reg == 0;
    }

    // warp reduce Z and V
    const auto __warp = ::cooperative_groups::tiled_partition<32, ::cooperative_groups::thread_block>(__group);
    ::cooperative_groups::reduce_update_async(
      __warp, __block_sum, __thread_sum, ::cooperative_groups::plus<__fp_type>());
    ::cooperative_groups::reduce_update_async(
      __warp, __block_zeroes, __thread_zeroes, ::cooperative_groups::plus<int>());
    __group.sync();

    if (__group.thread_rank() == 0)
    {
      const auto __z        = __block_sum.load(::cuda::std::memory_order_relaxed);
      const auto __v        = __block_zeroes.load(::cuda::std::memory_order_relaxed);
      const auto __finalize = ::cuda::experimental::cuco::__hyperloglog_ns::_Finalizer(__precision);
      __estimate            = __finalize(__z, __v);
    }
    __group.sync();

    return __estimate;
  }

  //! @brief Compute the estimated distinct items count.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @tparam _HostMemoryResource Host memory resource used for allocating the host buffer required to
  //! compute the final estimate by copying the sketch from device to host
  //!
  //! @param __host_mr Host memory resource used for copying the sketch
  //! @param __stream CUDA stream this operation is executed in
  //!
  //! @return Approximate distinct items count
  template <typename _HostMemoryResource>
  [[nodiscard]] _CCCL_HOST ::cuda::std::size_t
  __estimate(_HostMemoryResource __host_mr, ::cuda::stream_ref __stream) const
  {
    const auto __num_regs = __sketch.size();

    ::cuda::host_buffer<__register_type> __host_sketch_buf{__stream, __host_mr, __sketch.size(), ::cuda::no_init};

    ::cuda::__driver::__memcpyAsync(
      __host_sketch_buf.data(), __sketch.data(), sizeof(__register_type) * __num_regs, __stream.get());
    __stream.sync();

    __fp_type __sum = 0;
    int __zeroes    = 0;

    // geometric mean computation + count registers with 0s
    for (const auto __reg : __host_sketch_buf)
    {
      __sum += __fp_type{1} / static_cast<__fp_type>(1ull << __reg);
      __zeroes += __reg == 0;
    }

    const auto __finalize = ::cuda::experimental::cuco::__hyperloglog_ns::_Finalizer(__precision);

    // pass intermediate result to _Finalizer for bias correction, etc.
    return __finalize(__sum, __zeroes);
  }

  // #endif

  //! @brief Gets the hash function.
  //!
  //! @return The hash function
  [[nodiscard]] _CCCL_API constexpr auto __hash_function() const noexcept
  {
    return __hash;
  }

  //! @brief Gets the span of the sketch.
  //!
  //! @return The ::cuda::std::span of the sketch
  [[nodiscard]] _CCCL_API constexpr ::cuda::std::span<::cuda::std::byte> __sketch_span() const noexcept
  {
    return ::cuda::std::span<::cuda::std::byte>(reinterpret_cast<::cuda::std::byte*>(__sketch.data()), __sketch_bytes());
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] _CCCL_API constexpr ::cuda::std::size_t __sketch_bytes() const noexcept
  {
    return (1ull << __precision) * sizeof(__register_type);
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @param sketch_size_kb Upper bound sketch size in KB
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t
  __sketch_bytes(::cuda::experimental::cuco::__sketch_size_kb_t __sketch_size_kb) noexcept
  {
    // minimum precision is 4 or 64 bytes
    return ::cuda::std::max(static_cast<::cuda::std::size_t>(sizeof(__register_type) * (1ull << 4)),
                            ::cuda::std::bit_floor(static_cast<::cuda::std::size_t>(__sketch_size_kb * 1024)));
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @param __standard_deviation Upper bound standard deviation for approximation error
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t
  sketch_bytes(::cuda::experimental::cuco::__standard_deviation_t __standard_deviation) noexcept
  {
    // implementation taken from
    // https://github.com/apache/spark/blob/6a27789ad7d59cd133653a49be0bb49729542abe/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/util/HyperLogLogPlusPlusHelper.scala#L43

    auto const __precision_from_sd =
      static_cast<int>(::cuda::std::ceil(2.0 * ::cuda::std::log2(1.106 / __standard_deviation)));

    //  minimum precision is 4 or 64 bytes
    const auto __precision_ = ::cuda::std::max(static_cast<int>(4), __precision_from_sd);

    // inverse of this function (omitting the minimum precision constraint) is
    // standard_deviation = 1.106 / exp((__precision_ * log(2.0)) / 2.0)

    return sizeof(__register_type) * (1ull << __precision_);
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @param __precision HyperLogLog precision parameter
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t
  sketch_bytes(::cuda::experimental::cuco::__precision_t __precision) noexcept
  {
    const auto __precision_value = static_cast<int>(__precision);

    return sizeof(__register_type) * (1ull << __precision_value);
  }

  //! @brief Gets the alignment required for the sketch storage.
  //!
  //! @return The required alignment
  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t __sketch_alignment() noexcept
  {
    return alignof(__register_type);
  }

private:
  //!
  //! @brief Gets the register mask used to separate register index from count.
  //!
  //! @return The register mask
  [[nodiscard]] _CCCL_API constexpr __hash_value_type __register_mask() const noexcept
  {
    return (1ull << __precision) - 1;
  }

  //! @brief Atomically updates the register at position `i` with `max(reg[i], value)`.
  //!
  //! @param __i Register index
  //! @param __value New value
  _CCCL_DEVICE constexpr void __update_max(int __i, __register_type __value) noexcept
  {
    ::cuda::atomic_ref<__register_type, _Scope> __register_ref(__sketch[__i]);
    __register_ref.fetch_max(__value, ::cuda::memory_order_relaxed);
  }

  //! @brief Try expanding the shmem partition for a given kernel beyond 48KB if necessary.
  //!
  //! @tparam _Kernel Type of kernel function
  //!
  //! @param __kernel The kernel function
  //! @param __shmem_bytes Number of requested dynamic shared memory bytes
  //!
  //! @returns True iff kernel configuration is successful
  template <typename _Kernel>
  [[nodiscard]] _CCCL_HOST constexpr bool __try_reserve_shmem(_Kernel __kernel, int __shmem_bytes) const
  {
    int __device = -1;
    _CCCL_TRY_CUDA_API(::cudaGetDevice, "cudaGetDevice failed", &__device);
    int __max_shmem_bytes = 0;
    _CCCL_TRY_CUDA_API(
      ::cudaDeviceGetAttribute,
      "cudaDeviceGetAttribute failed",
      &__max_shmem_bytes,
      ::cudaDevAttrMaxSharedMemoryPerBlockOptin,
      __device);

    if (__shmem_bytes <= __max_shmem_bytes)
    {
      _CCCL_TRY_CUDA_API(
        ::cudaFuncSetAttribute,
        "cudaFuncSetAttribute failed",
        reinterpret_cast<const void*>(__kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        __shmem_bytes);
      return true;
    }
    else
    {
      return false;
    }
  }
};
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___HYPERLOGLOG_IMPL_CUH
