//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//                         Copyright (c) 2023 Maikel Nadolski
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_ATOMIC_INTRUSIVE_QUEUE
#define __CUDAX_EXECUTION_ATOMIC_INTRUSIVE_QUEUE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/atomic>

#include <cuda/experimental/__execution/intrusive_queue.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
// An atomic queue that supports multiple producers and a single consumer.
template <auto _NextPtr>
class _CCCL_TYPE_VISIBILITY_DEFAULT __atomic_intrusive_queue;

template <class _Tp, _Tp* _Tp::* _NextPtr>
class alignas(64) __atomic_intrusive_queue<_NextPtr>
{
public:
  _CCCL_API auto push(_Tp* __node) noexcept -> bool
  {
    _CCCL_ASSERT(__node != nullptr, "Cannot push a null pointer to the queue");
    _Tp* __old_head = __head_.load(::cuda::std::memory_order_relaxed);
    do
    {
      __node->*_NextPtr = __old_head;
    } while (!__head_.compare_exchange_weak(__old_head, __node, ::cuda::std::memory_order_acq_rel));

    // If the queue was empty before, we notify the consumer thread that there is now an
    // item available. If the queue was not empty, we do not notify, because the consumer
    // thread has already been notified.
    if (__old_head != nullptr)
    {
      return false;
    }

    // There can be only one consumer thread, so we can use notify_one here instead of
    // notify_all:
    __head_.notify_one();
    return true;
  }

  _CCCL_API void wait_for_item() noexcept
  {
    // Wait until the queue has an item in it:
    __head_.wait(nullptr);
  }

  [[nodiscard]]
  _CCCL_API auto pop_all() noexcept -> __intrusive_queue<_NextPtr>
  {
    auto* const __list = __head_.exchange(nullptr, ::cuda::std::memory_order_acquire);
    return __intrusive_queue<_NextPtr>::make_reversed(__list);
  }

private:
  ::cuda::std::atomic<_Tp*> __head_{nullptr};
};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_ATOMIC_INTRUSIVE_QUEUE
