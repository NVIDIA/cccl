//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_OLD_THREAD_GROUP_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_OLD_THREAD_GROUP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/all_devices.h>
#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__barrier/barrier.h>
#include <cuda/std/__cstddef/byte.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__host_stdlib/memory>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__memory/align.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/atomic>
#include <cuda/std/memory>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
class thread_group
{
  static constexpr ::cuda::std::byte __MAGIC_INIT_VALUE{222};
  static constexpr ::cuda::std::byte __MAGIC_CONSTRUCTED_VALUE{0};

  // Otherwise the construction signaling doesn't work
  static_assert(__MAGIC_INIT_VALUE != __MAGIC_CONSTRUCTED_VALUE);

  class __shared_data_t
  {
    class __payload_t
    {
    public:
      _CCCL_HOST_API explicit __payload_t(const ::cuda::std::uint32_t __group_size,
                                          const ::cuda::std::span<::cuda::std::byte> __mem)
          // Do not initialize __constructed_flag in initializer list, it must be initialized
          // atomically
          : __ref_count_{__group_size}
          , __bar_{static_cast<::cuda::std::ptrdiff_t>(__group_size)}
          , __scratch_mem_{__mem}
      {
        static_assert(offsetof(__payload_t, __constructed_flag_) == 0);
        const auto __ref = ::cuda::std::atomic_ref{__constructed_flag_};

        __ref.store(__MAGIC_CONSTRUCTED_VALUE, ::cuda::std::memory_order_release);
        __ref.notify_all();
      }

      [[nodiscard]] _CCCL_HOST_API static ::std::shared_ptr<::cuda::std::byte[]> __make_payload(
        const ::cuda::std::uint32_t __rank,
        const ::cuda::std::uint32_t __size,
        ::std::shared_ptr<::cuda::std::byte[]> __shared_mem)
      {
        // Technically only thread 0 needs to perform any of the pointer aligning, but we let all
        // threads do it so we can error-check the arguments properly.
        auto* const __raw_ptr = __shared_mem.get();
        auto* __scratch_ptr   = static_cast<void*>(__raw_ptr + sizeof(__payload_t));
        auto __capacity       = required_shared_memory_size(__size) - sizeof(__payload_t);

        if (!::cuda::std::align(
              /*__alignment*/ alignof(::cuda::std::max_align_t),
              /*__size*/ __required_scratch_mem_size(__size),
              __scratch_ptr,
              __capacity))
        {
          _CCCL_THROW(::std::length_error,
                      "Failed to align scratch memory pointer. Likely allocated shared memory is insufficient.");
        }

        // Construct last, to preserve strong exception guarantee.
        if (__rank == 0)
        {
          auto* const __ptr = reinterpret_cast<__payload_t*>(__raw_ptr);
          auto __span =
            ::cuda::std::span<::cuda::std::byte>{static_cast<::cuda::std::byte*>(__scratch_ptr), __capacity};

          ::cuda::std::__construct_at(__ptr, __size, __span);
          _CCCL_VERIFY(&__ptr->__constructed_flag_ == __raw_ptr,
                       "__payload constructed at an offset inside shared memory scratch buffer");
        }
        else
        {
          const auto __ref = ::cuda::std::atomic_ref{__shared_mem[0]};

          __ref.wait(__MAGIC_INIT_VALUE, ::cuda::std::memory_order_acquire);
        }
        return __shared_mem;
      }

      // This member exists so that the other threads in the group can detect when the first
      // thread successfully constructs the shared data. We must need to use a bare
      // cuda::std::byte (accessed via atomic refs) because the other threads can only safely
      // access the buffer as raw bytes. I would like to have used `atomic<byte>`, but we have
      // no guarantee on the placement of the byte within the atomic, so raw byte it is.
      //
      // We need the __constructed_flag member to sit at exactly the first byte of the shared
      // memory buffer (not __scratch_mem, the original buffer).
      //
      // If this value is ever not at the very first byte, then we will write to the wrong
      // memory location and never release the waiting threads.
      ::cuda::std::byte __constructed_flag_;
      ::cuda::std::atomic<::cuda::std::uint32_t> __ref_count_{};
      ::cuda::std::barrier<> __bar_;
      ::cuda::std::span<::cuda::std::byte> __scratch_mem_{};
    };

  public:
    _CCCL_HIDE_FROM_ABI constexpr __shared_data_t() = default;

    _CCCL_HOST_API __shared_data_t(const ::cuda::std::uint32_t __rank,
                                   const ::cuda::std::uint32_t __size,
                                   ::std::shared_ptr<::cuda::std::byte[]> __shared_mem)
        : __mem_{__payload_t::__make_payload(__rank, __size, ::cuda::std::move(__shared_mem))}
    {}

    _CCCL_HOST_API __shared_data_t(const __shared_data_t& __other) noexcept
        : __mem_{__other.__mem_}
    {
      if (auto* const __p = __payload())
      {
        __p->__ref_count_.fetch_add(/*__op=*/1, ::cuda::std::memory_order_relaxed);
      }
    }

    _CCCL_HOST_API __shared_data_t& operator=(const __shared_data_t& __other) noexcept
    {
      __shared_data_t{__other}.swap(*this);
      return *this;
    }

    // We can = default this because we never have a payload
    _CCCL_HIDE_FROM_ABI __shared_data_t(__shared_data_t&& __other) noexcept = default;

    _CCCL_HIDE_FROM_ABI __shared_data_t& operator=(__shared_data_t&& __other) noexcept
    {
      // Note, we cannot = default this, because we may have a payload prior to assignment. In
      // that case, it needs to be disposed of properly (which the temporary below will do).
      __shared_data_t{::cuda::std::move(__other)}.swap(*this);
      return *this;
    }

    _CCCL_HIDE_FROM_ABI ~__shared_data_t() noexcept
    {
      __reset();
    }

    _CCCL_HOST_API void swap(__shared_data_t& __other) noexcept
    {
      __mem_.swap(__other.__mem_);
    }

    [[nodiscard]] _CCCL_HOST_API ::cuda::std::barrier<>& __barrier() noexcept
    {
      return __payload()->__bar_;
    }

    [[nodiscard]] _CCCL_HOST_API const ::cuda::std::barrier<>& __barrier() const noexcept
    {
      return __payload()->__bar_;
    }

    template <typename _Tp>
    [[nodiscard]] _CCCL_HOST_API ::cuda::std::span<_Tp>
    __scratch_mem_as(::cuda::std::uint32_t __group_size) const noexcept
    {
      const auto __required_size = __payload()->__scratch_mem_.size_bytes() / sizeof(_Tp);

      // required_scratch_mem_size() is too small
      _CCCL_VERIFY(__required_size >= __group_size,
                   "Recast scratch memory too small to hold entire groups private memory");

      auto* const __ptr = ::cuda::std::launder(reinterpret_cast<_Tp*>(__payload()->__scratch_mem_.data()));

      return cuda::std::span<_Tp>{__ptr, __group_size};
    }

    [[nodiscard]] _CCCL_HOST_API static constexpr ::cuda::std::size_t
    __required_shared_memory_size(::cuda::std::uint32_t __group_size) noexcept
    {
      return sizeof(__payload_t) + alignof(__payload_t) - 1 + __required_scratch_mem_size(__group_size);
    }

  private:
    [[nodiscard]] _CCCL_HOST_API static constexpr ::cuda::std::size_t
    __required_scratch_mem_size(::cuda::std::uint32_t __group_size) noexcept
    {
      // The scratch buffer is used to communicate between threads, so it should be large
      // enough that each thread can write its complete payload into its private area. Since we
      // cannot grow the scratch buffer, once allocated we need to pick a number upfront that
      // is large enough to hold any reasonable payload.
      return __group_size * (4 * sizeof(void*));
    }

    [[nodiscard]] _CCCL_HOST_API __payload_t* __payload() noexcept
    {
      return reinterpret_cast<__payload_t*>(__mem_.get());
    }

    [[nodiscard]] _CCCL_HOST_API const __payload_t* __payload() const noexcept
    {
      return reinterpret_cast<const __payload_t*>(__mem_.get());
    }

    _CCCL_HOST_API void __reset() noexcept
    {
      // We could have been moved-from, so need to test __p
      if (auto* const __p = __payload())
      {
        // There is a small micro-optimization we can employ here. Normally, reference count
        // decrements need to be memory_order_acq_rel or memory_order_seq_cst because the last
        // thread to decrement also needs to see all the write effects of prior threads. We can,
        // however, relax the decrements to memory_order_release, so long as we acquire before
        // making any changes.
        if (__p->__ref_count_.fetch_sub(/*__op=*/1, ::cuda::std::memory_order_release) - 1 == 0)
        {
          // Hence this threadfence.
          ::cuda::std::atomic_thread_fence(::cuda::std::memory_order_acquire);
          ::cuda::std::destroy_at(__p);
        }
        __mem_.reset();
      }
    }

    ::std::shared_ptr<::cuda::std::byte[]> __mem_{};
  };

  struct __private_tag
  {};

public:
  _CCCL_HOST_API thread_group(const ::cuda::__all_devices& __devices)
      : thread_group{__singleton_all_local_devices(__devices)}
  {}

  _CCCL_HOST_API thread_group(const ::cuda::std::uint32_t __rank,
                              const ::cuda::std::uint32_t __group_size,
                              ::std::shared_ptr<::cuda::std::byte[]> __shared_mem)
      : thread_group{__private_tag{}, __rank, __group_size, ::cuda::std::move(__shared_mem)}

  {}

  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::uint32_t rank() const noexcept
  {
    return __rank_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::uint32_t size() const noexcept
  {
    return __group_size_;
  }

  _CCCL_HOST_API void barrier() const noexcept
  {
    if (size() > 1)
    {
      __mut_shared_data().__barrier().arrive_and_wait();
    }
  }

  _CCCL_TEMPLATE(typename _Tp, typename _Range)
  _CCCL_REQUIRES(::cuda::std::ranges::output_range<_Range, _Tp>)
  void all_gather(_Tp __data, _Range&& __dest)
  {
    if constexpr (::cuda::std::ranges::sized_range<_Range>)
    {
      _CCCL_VERIFY(::cuda::std::ranges::size(__dest) >= size(),
                   "Destination buffer too small, must be at least of size group.size()");
    }

    auto __scratch = __shared_data().__scratch_mem_as<_Tp>(size());

    ::cuda::std::uninitialized_move_n(::cuda::std::addressof(__data), 1, __scratch.begin() + rank());
    barrier();
    ::cuda::std::copy(__scratch.begin(), __scratch.end(), ::cuda::std::ranges::begin(__dest));
    barrier();
    if (rank() == 0)
    {
      ::cuda::std::destroy(__scratch.begin(), __scratch.end());
    }
    // We need to ensure rank 0 has properly destroyed the scratch space to ensure subsequent
    // calls to this function don't see a partially destroyed scratch buffer. We could either
    // have a barrier in the beginning of the function or here, but we choose to do it here
    // since the threads are all "together" at this point anyways.
    barrier();
  }

  _CCCL_TEMPLATE(typename _Tp, typename _Range)
  _CCCL_REQUIRES(::cuda::std::ranges::output_range<_Range, _Tp>)
  void gather(::cuda::std::uint32_t __root, _Tp __data, _Range&& __dest)
  {
    const auto __is_target_rank = rank() == __root;

    _CCCL_VERIFY(__root >= 0 && __root < size(), "Invalid root rank not in range for current group");
    if constexpr (::cuda::std::ranges::sized_range<_Range>)
    {
      // Size of the range is only relevant on the root
      if (__is_target_rank)
      {
        _CCCL_VERIFY(::cuda::std::ranges::size(__dest) >= size(),
                     "Destination buffer too small, must be at least of size group.size()");
      }
    }

    auto __scratch = __shared_data().__scratch_mem_as<_Tp>(size());

    ::cuda::std::uninitialized_move_n(::cuda::std::addressof(__data), 1, __scratch.begin() + rank());
    barrier();
    if (__is_target_rank)
    {
      ::cuda::std::move(__scratch.begin(), __scratch.end(), ::cuda::std::ranges::begin(__dest));
      ::cuda::std::destroy(__scratch.begin(), __scratch.end());
    }
    // We need to ensure the target has properly destroyed the scratch space to ensure
    // subsequent calls to this function don't see a partially destroyed scratch buffer. We
    // could either have a barrier in the beginning of the function or here, but we choose to
    // do it here since the threads are all "together" at this point anyways.
    barrier();
  }

  _CCCL_TEMPLATE(typename _Tp, typename _Range)
  _CCCL_REQUIRES(::cuda::std::ranges::output_range<_Range, _Tp>)
  void bcast(::cuda::std::uint32_t __root, _Tp __data, _Range&& __dest)
  {
    const auto __is_bcast_rank = rank() == __root;

    _CCCL_VERIFY(__root >= 0 && __root < size(), "Invalid root rank not in range for current group");
    if constexpr (::cuda::std::ranges::sized_range<_Range>)
    {
      _CCCL_VERIFY(::cuda::std::ranges::size(__dest) >= size(),
                   "Destination buffer too small, must be at least of size group.size()");
    }

    auto __scratch = __shared_data().__scratch_mem_as<_Tp>(/*__group_size*/ 1);

    if (__is_bcast_rank)
    {
      ::cuda::std::uninitialized_move_n(::cuda::std::addressof(__data), 1, __scratch.begin());
    }
    barrier();
    ::cuda::std::copy(__scratch.begin(), __scratch.begin() + 1, ::cuda::std::ranges::begin(__dest));
    barrier();
    if (__is_bcast_rank)
    {
      ::cuda::std::destroy(__scratch.begin(), __scratch.begin() + 1);
    }
    // We need to ensure that the bcasting rank has properly destroyed the scratch space to
    // ensure subsequent calls to this function don't see a partially destroyed scratch
    // buffer. We could either have a barrier in the beginning of the function or here, but we
    // choose to do it here since the threads are all "together" at this point anyways.
    barrier();
  }

  [[nodiscard]] _CCCL_HOST_API static constexpr ::cuda::std::size_t
  required_shared_memory_size(::cuda::std::uint32_t __group_size) noexcept
  {
    return __shared_data_t::__required_shared_memory_size(__group_size);
  }

  _CCCL_HOST_API static constexpr void* initialize_shared_memory(void* __mem) noexcept
  {
    // If we are in a consteval context then a simple static_cast suffices, but otherwise we
    // must atomically initialize. It is possible that the user may be using one of the
    // participating threads to initialize the memory.
    _CCCL_IF_CONSTEVAL
    {
      *static_cast<::cuda::std::byte*>(__mem) = __MAGIC_INIT_VALUE;
    }
    else
    {
      ::cuda::std::atomic_ref{*static_cast<::cuda::std::byte*>(__mem)}.store(
        __MAGIC_INIT_VALUE, ::cuda::std::memory_order_seq_cst);
    }
    return __mem;
  }

private:
  [[nodiscard]] _CCCL_HOST_API static const thread_group& __singleton_all_local_devices(const ::cuda::__all_devices&)
  {
    static const auto __ret = [&] {
      constexpr auto __group_size = 1;
      auto __mem =
        ::std::shared_ptr<::cuda::std::byte[]>{new ::cuda::std::byte[required_shared_memory_size(__group_size)]};

      initialize_shared_memory(__mem.get());

      return thread_group{/*__rank*/ 0, __group_size, ::cuda::std::move(__mem)};
    }();

    return __ret;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr __shared_data_t& __shared_data() noexcept
  {
    return __shared_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr const __shared_data_t& __shared_data() const noexcept
  {
    return __shared_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr __shared_data_t& __mut_shared_data() const noexcept
  {
    return const_cast<thread_group&>(*this).__shared_data();
  }

  // Grand central constructor, all other constructors must come through this one
  _CCCL_HOST_API thread_group(const __private_tag,
                              const ::cuda::std::uint32_t __rank,
                              const ::cuda::std::uint32_t __group_size,
                              ::std::shared_ptr<::cuda::std::byte[]> __shared_mem)
      : __rank_{[&] {
        _CCCL_VERIFY(__group_size > 0, "Group size must be > 0");
        _CCCL_VERIFY(__rank < __group_size, "Rank out of range of group size");
        _CCCL_VERIFY(__shared_mem.get() != nullptr, "Shared memory must not be NULL");
        return __rank;
      }()}
      , __group_size_{__group_size}
      , __shared_{__rank, __group_size, ::cuda::std::move(__shared_mem)}
  {}

  ::cuda::std::uint32_t __rank_{};
  ::cuda::std::uint32_t __group_size_{};
  __shared_data_t __shared_{};
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

// NOLINTEND(bugprone-reserved-identifier)

#endif // _CUDA_EXPERIMENTAL___THREAD_GROUP_THREAD_GROUP_H
