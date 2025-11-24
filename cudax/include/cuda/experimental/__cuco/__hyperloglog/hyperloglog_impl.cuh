//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__CUCO__HYPERLOGLOG__IMPL_CUH
#define _CUDAX__CUCO__HYPERLOGLOG__IMPL_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/__runtime/api_wrapper.h>
#include <cuda/atomic>
#include <cuda/std/__algorithm/max.h> // TODO #include <cuda/std/algorithm> once available
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/span>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/__cuco/__hyperloglog/finalizer.cuh>
#include <cuda/experimental/__cuco/__hyperloglog/kernels.cuh>
#include <cuda/experimental/__cuco/__utility/strong_type.cuh>
#include <cuda/experimental/__cuco/hash_functions.cuh>

#include <vector>

#include <cooperative_groups.h>

#include <cooperative_groups/reduce.h>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
CUDAX_CUCO_DEFINE_STRONG_TYPE(__sketch_size_kb_t, double);

CUDAX_CUCO_DEFINE_STRONG_TYPE(__standard_deviation_t, double);

//! @brief A GPU-accelerated utility for approximating the number of distinct items in a multiset.
//!
//! @note This class implements the HyperLogLog/HyperLogLog++ algorithm:
//! https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf.
//!
//! @tparam _Tp Type of items to count
//! @tparam _Scope The scope in which operations will be performed by individual threads
//! @tparam _Hash Hash function used to hash items
template <class _Tp, ::cuda::thread_scope _Scope, class _Hash>
class _HyperLogLog_Impl
{
  // We use `int` here since this is the smallest type that supports native `atomicMax` on GPUs
  using __fp_type         = double; ///< Floating point type used for reduction
  using __hash_value_type = decltype(::cuda::std::declval<_Hash>()(::cuda::std::declval<_Tp>())); ///< Hash value type

public:
  static constexpr auto thread_scope = _Scope; ///< CUDA thread scope

  using value_type    = _Tp; ///< Type of items to count
  using hasher        = _Hash; ///< Hash function type
  using register_type = int; ///< HLL register type

  template <::cuda::thread_scope _NewScope>
  using with_scope = _HyperLogLog_Impl<_Tp, _NewScope, _Hash>; ///< Ref type with different thread scope

  //! @brief Constructs a non-owning `_HyperLogLog_Impl` object.
  //!
  //! @throw If sketch size < 0.0625KB or 64B or standard deviation > 0.2765. Throws if called from
  //! host; UB if called from device.
  //! @throw If sketch storage has insufficient alignment. Throws if called from host; UB if called.
  //! from device.
  //!
  //! @param sketch_span Reference to sketch storage
  //! @param hash The hash function used to hash items
  _CCCL_API constexpr _HyperLogLog_Impl(::cuda::std::span<::cuda::std::byte> sketch_span, const _Hash& hash)
      : __hash{hash}
      , __precision{::cuda::std::countr_zero(
          __sketch_bytes(static_cast<::cuda::experimental::cuco::__sketch_size_kb_t>(sketch_span.size() / 1024.0))
          / sizeof(register_type))}
      , __register_mask{(1ull << this->__precision) - 1}
      , __sketch{reinterpret_cast<register_type*>(sketch_span.data()), this->__sketch_bytes() / sizeof(register_type)}
  {
#ifndef __CUDA_ARCH__
    const auto __alignment =
      1ull << ::cuda::std::countr_zero(reinterpret_cast<::cuda::std::uintptr_t>(sketch_span.data()));
    _CCCL_ASSERT(__alignment >= __sketch_alignment(), "Insufficient sketch alignment");

    _CCCL_ASSERT(this->__precision >= 4, "Minimum required sketch size is 0.0625KB or 64B");
#endif
  }

  //! @brief Resets the estimator, i.e., clears the current count estimate.
  //!
  //! @tparam _CG CUDA Cooperative Group type
  //!
  //! @param __group CUDA Cooperative group this operation is executed in
  template <class _CG>
  _CCCL_DEVICE constexpr void __clear(_CG __group) noexcept
  {
    for (int __i = __group.thread_rank(); __i < this->__sketch.size(); __i += __group.size())
    {
      new (&(this->__sketch[__i])) register_type{};
    }
  }

  //! @brief Resets the estimator, i.e., clears the current count estimate.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `__clear_async`.
  //!
  //! @param stream CUDA stream this operation is executed in
  _CCCL_HOST constexpr void __clear(::cuda::stream_ref __stream)
  {
    this->__clear_async(__stream);
    __stream.sync();
  }

  //! @brief Asynchronously resets the estimator, i.e., clears the current count estimate.
  //!
  //! @param stream CUDA stream this operation is executed in
  _CCCL_HOST constexpr void __clear_async(::cuda::stream_ref __stream)
  {
    auto constexpr __block_size = 1024;
    ::cuda::experimental::cuco::__hyperloglog_ns::__clear<<<1, __block_size, 0, __stream.get()>>>(*this);
  }

  //! @brief Adds an item to the estimator.
  //!
  //! @param item The item to be counted
  _CCCL_DEVICE constexpr void __add(const _Tp& __item) noexcept
  {
    const auto __h      = this->__hash(__item);
    const auto __reg    = __h & this->__register_mask;
    const auto __zeroes = ::cuda::std::countl_zero(__h | this->__register_mask) + 1; // __clz

    // reversed order (same one as Spark uses)
    // const auto __reg    = __h >> ((sizeof(__hash_value_type) * 8) - this->__precision);
    // const auto __zeroes = ::cuda::std::countl_zero(__h << this->__precision) + 1;

    this->__update_max(__reg, __zeroes);
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
      const auto __ptr                  = thrust::raw_pointer_cast(&__first[0]);
      auto constexpr __max_vector_bytes = 32;
      const auto __alignment =
        1 << ::cuda::std::countr_zero(reinterpret_cast<::cuda::std::uintptr_t>(__ptr) | __max_vector_bytes);
      const auto __vector_size = __alignment / sizeof(value_type);

      switch (__vector_size)
      {
        case 2:
          __kernel = reinterpret_cast<const void*>(
            ::cuda::experimental::cuco::__hyperloglog_ns::__add_shmem_vectorized<2, _HyperLogLog_Impl>);
          break;
        case 4:
          __kernel = reinterpret_cast<const void*>(
            ::cuda::experimental::cuco::__hyperloglog_ns::__add_shmem_vectorized<4, _HyperLogLog_Impl>);
          break;
        case 8:
          __kernel = reinterpret_cast<const void*>(
            ::cuda::experimental::cuco::__hyperloglog_ns::__add_shmem_vectorized<8, _HyperLogLog_Impl>);
          break;
        case 16:
          __kernel = reinterpret_cast<const void*>(
            ::cuda::experimental::cuco::__hyperloglog_ns::__add_shmem_vectorized<16, _HyperLogLog_Impl>);
          break;
      };
    }

    if (__kernel != nullptr and this->__try_reserve_shmem(__kernel, __shmem_bytes))
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

        const auto __ptr      = thrust::raw_pointer_cast(&__first[0]);
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
        ::cuda::experimental::cuco::__hyperloglog_ns::__add_shmem<_InputIt, _HyperLogLog_Impl>);
      void* __kernel_args[] = {const_cast<void*>(reinterpret_cast<const void*>(&__first)),
                               const_cast<void*>(reinterpret_cast<const void*>(&__num_items)),
                               reinterpret_cast<void*>(this)};
      if (this->__try_reserve_shmem(__kernel, __shmem_bytes))
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
          ::cuda::experimental::cuco::__hyperloglog_ns::__add_gmem<_InputIt, _HyperLogLog_Impl>);

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
    this->__add_async(__first, __last, __stream);
    __stream.sync();
  }

  //! @brief Merges the result of `other` estimator reference into `*this` estimator reference.
  //!
  //! @throw If this->__sketch_bytes() != other.__sketch_bytes() then behavior is undefined
  //!
  //! @tparam _CG CUDA Cooperative Group type
  //! @tparam _OtherScope Thread scope of `other` estimator
  //!
  //! @param __group CUDA Cooperative group this operation is executed in
  //! @param __other Other estimator reference to be merged into `*this`
  template <class _CG, ::cuda::thread_scope _OtherScope>
  _CCCL_DEVICE constexpr void __merge(_CG __group, _HyperLogLog_Impl<_Tp, _OtherScope, _Hash>& __other)
  {
    // TODO find a better way to do error handling in device code
    // if (__other.__precision != this->__precision) { __trap(); }

    for (int __i = __group.thread_rank(); __i < this->__sketch.size(); __i += __group.size())
    {
      this->__update_max(__i, __other.__sketch[__i]);
    }
  }

#if 0
  //! @brief Asynchronously merges the result of `other` estimator reference into `*this`
  //! estimator.
  //!
  //! @throw If this->sketch_bytes() != other.sketch_bytes()
  //!
  //! @tparam _OtherScope Thread scope of `other` estimator
  //!
  //! @param other Other estimator reference to be merged into `*this`
  //! @param stream CUDA stream this operation is executed in
  template <::cuda::thread_scope _OtherScope>
  _CCCL_HOST constexpr void merge_async(_HyperLogLog_Impl<_Tp, _OtherScope, const _Hash>& __other, ::cuda::stream_ref __stream)
  {
    CUCO_EXPECTS(__other.__precision == this->__precision, "Cannot merge estimators with different sketch sizes");
    auto constexpr __block_size = 1024;
    ::cuda::experimental::cuco::__hyperloglog_ns::__merge<<<1, __block_size, 0, __stream.get()>>>(__other, *this);
  }

  //! @brief Merges the result of `other` estimator reference into `*this` estimator.
  //!
  //! @note This function synchronizes the given stream. For asynchronous execution use
  //! `merge_async`.
  //!
  //! @throw If this->sketch_bytes() != other.sketch_bytes()
  //!
  //! @tparam _OtherScope Thread scope of `other` estimator
  //!
  //! @param other Other estimator reference to be merged into `*this`
  //! @param stream CUDA stream this operation is executed in
  template <::cuda::thread_scope _OtherScope>
  _CCCL_HOST constexpr void merge(_HyperLogLog_Impl<_Tp, _OtherScope, const _Hash>& __other, ::cuda::stream_ref __stream)
  {
    this->merge_async(__other, __stream);
    __stream.sync();
  }
#endif

  //! @brief Compute the estimated distinct items count.
  //!
  //! @param group CUDA thread block group this operation is executed in
  //!
  //! @return Approximate distinct items count
  [[nodiscard]] _CCCL_DEVICE size_t __estimate(const cooperative_groups::thread_block& __group) const noexcept
  {
    __shared__ ::cuda::atomic<__fp_type, ::cuda::std::thread_scope_block> __block_sum;
    __shared__ ::cuda::atomic<int, ::cuda::std::thread_scope_block> __block_zeroes;
    __shared__ size_t __estimate;

    if (__group.thread_rank() == 0)
    {
      new (&__block_sum) decltype(__block_sum){0};
      new (&__block_zeroes) decltype(__block_zeroes){0};
    }
    __group.sync();

    __fp_type __thread_sum = 0;
    int __thread_zeroes    = 0;
    for (int __i = __group.thread_rank(); __i < this->__sketch.size(); __i += __group.size())
    {
      const auto __reg = this->__sketch[__i];
      __thread_sum += __fp_type{1} / static_cast<__fp_type>(1 << __reg);
      __thread_zeroes += __reg == 0;
    }

    // warp reduce Z and V
    const auto __warp = cooperative_groups::tiled_partition<32, cooperative_groups::thread_block>(__group);

    // TODO check if this is always true with latest ctk or cccl version and remove
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
    cooperative_groups::reduce_update_async(__warp, __block_sum, __thread_sum, cooperative_groups::plus<__fp_type>());
    cooperative_groups::reduce_update_async(__warp, __block_zeroes, __thread_zeroes, cooperative_groups::plus<int>());
#else
    const auto __warp_sum    = cooperative_groups::reduce(__warp, __thread_sum, cooperative_groups::plus<__fp_type>());
    const auto __warp_zeroes = cooperative_groups::reduce(__warp, __thread_zeroes, cooperative_groups::plus<int>());
    // TODO warp sync needed?
    // TODO use invoke_one
    if (__warp.thread_rank() == 0)
    {
      __block_sum.fetch_add(__warp_sum, ::cuda::std::memory_order_relaxed);
      __block_zeroes.fetch_add(__warp_zeroes, ::cuda::std::memory_order_relaxed);
    }
#endif
    __group.sync();

    if (__group.thread_rank() == 0)
    {
      const auto __z        = __block_sum.load(::cuda::std::memory_order_relaxed);
      const auto __v        = __block_zeroes.load(::cuda::std::memory_order_relaxed);
      const auto __finalize = ::cuda::experimental::cuco::__hyperloglog_ns::_Finalizer(this->__precision);
      __estimate            = __finalize(__z, __v);
    }
    __group.sync();

    return __estimate;
  }

  //! @brief Compute the estimated distinct items count.
  //!
  //! @note This function synchronizes the given stream.
  //!
  //! @param stream CUDA stream this operation is executed in
  //!
  //! @return Approximate distinct items count
  [[nodiscard]] _CCCL_HOST size_t __estimate(::cuda::stream_ref __stream) const
  {
    const auto __num_regs = 1ull << this->__precision;
    ::std::vector<register_type> __host_sketch(__num_regs);

    // TODO check if storage is host accessible
    _CCCL_TRY_CUDA_API(
      ::cudaMemcpyAsync,
      "cudaMemcpyAsync failed",
      __host_sketch.data(),
      this->__sketch.data(),
      sizeof(register_type) * __num_regs,
      cudaMemcpyDefault,
      __stream.get());

    __stream.sync();

    __fp_type __sum = 0;
    int __zeroes    = 0;

    // geometric mean computation + count registers with 0s
    for (const auto __reg : __host_sketch)
    {
      __sum += __fp_type{1} / static_cast<__fp_type>(1ull << __reg);
      __zeroes += __reg == 0;
    }

    const auto __finalize = ::cuda::experimental::cuco::__hyperloglog_ns::_Finalizer(this->__precision);

    // pass intermediate result to _Finalizer for bias correction, etc.
    return __finalize(__sum, __zeroes);
  }

  // #endif

  //! @brief Gets the hash function.
  //!
  //! @return The hash function
  [[nodiscard]] _CCCL_API constexpr auto __hash_function() const noexcept
  {
    return this->__hash;
  }

  //! @brief Gets the span of the sketch.
  //!
  //! @return The ::cuda::std::span of the sketch
  [[nodiscard]] _CCCL_API constexpr ::cuda::std::span<::cuda::std::byte> __sketch_span() const noexcept
  {
    return ::cuda::std::span<::cuda::std::byte>(
      reinterpret_cast<::cuda::std::byte*>(this->__sketch.data()), this->__sketch_bytes());
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] _CCCL_API constexpr size_t __sketch_bytes() const noexcept
  {
    return (1ull << this->__precision) * sizeof(register_type);
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @param sketch_size_kb Upper bound sketch size in KB
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] _CCCL_API static constexpr size_t
  __sketch_bytes(::cuda::experimental::cuco::__sketch_size_kb_t __sketch_size_kb) noexcept
  {
    // minimum precision is 4 or 64 bytes
    return ::cuda::std::max(static_cast<size_t>(sizeof(register_type) * 1ull << 4),
                            ::cuda::std::bit_floor(static_cast<size_t>(__sketch_size_kb * 1024)));
  }

  //! @brief Gets the number of bytes required for the sketch storage.
  //!
  //! @param __standard_deviation Upper bound standard deviation for approximation error
  //!
  //! @return The number of bytes required for the sketch
  [[nodiscard]] _CCCL_API static constexpr std::size_t
  sketch_bytes(::cuda::experimental::cuco::__standard_deviation_t __standard_deviation) noexcept
  {
    // implementation taken from
    // https://github.com/apache/spark/blob/6a27789ad7d59cd133653a49be0bb49729542abe/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/util/HyperLogLogPlusPlusHelper.scala#L43

    //  minimum precision is 4 or 64 bytes
    const auto __precision_ = ::cuda::std::max(
      static_cast<int32_t>(4),
      static_cast<int32_t>(
        ::cuda::std::ceil(2.0 * ::cuda::std::log(1.106 / __standard_deviation) / ::cuda::std::log(2.0))));

    // inverse of this function (omitting the minimum precision constraint) is
    // standard_deviation = 1.106 / exp((__precision_ * log(2.0)) / 2.0)

    return sizeof(register_type) * (1ull << __precision_);
  }

  //! @brief Gets the alignment required for the sketch storage.
  //!
  //! @return The required alignment
  [[nodiscard]] _CCCL_API static constexpr size_t __sketch_alignment() noexcept
  {
    return alignof(register_type);
  }

private:
  //! @brief Atomically updates the register at position `i` with `max(reg[i], value)`.
  //!
  //! @tparam Scope CUDA thread scope
  //!
  //! @param i Register index
  //! @param value New value
  _CCCL_DEVICE constexpr void __update_max(int __i, register_type __value) noexcept
  {
    ::cuda::atomic_ref<register_type, _Scope> __register_ref(this->__sketch[__i]);
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
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
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

  hasher __hash; ///< Hash function used to hash items
  int32_t __precision; ///< HLL precision parameter
  __hash_value_type __register_mask; ///< Mask used to separate register index from count
  ::cuda::std::span<register_type> __sketch; ///< HLL sketch storage

  template <class _Tp_, ::cuda::thread_scope _Scope_, class _Hash_>
  friend struct _HyperLogLog_Impl;
};
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO__HYPERLOGLOG__IMPL_CUH
