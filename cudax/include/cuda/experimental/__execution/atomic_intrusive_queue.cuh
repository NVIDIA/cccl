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
template <auto _NextPtr>
class _CCCL_TYPE_VISIBILITY_DEFAULT __atomic_intrusive_queue;

template <class _Tp, _Tp* _Tp::* _NextPtr>
class alignas(64) __atomic_intrusive_queue<_NextPtr>
{
public:
  using __node_pointer _CCCL_NODEBUG_ALIAS        = _Tp*;
  using __atomic_node_pointer _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::atomic<_Tp*>;

  [[nodiscard]]
  _CCCL_API auto empty() const noexcept -> bool
  {
    return __head_.load(_CUDA_VSTD::memory_order_relaxed) == nullptr;
  }

  _CCCL_API auto push(__node_pointer __node) noexcept -> bool
  {
    _CCCL_ASSERT(__node != nullptr, "Cannot push a null pointer to the queue");
    __node_pointer __old_head = __head_.load(_CUDA_VSTD::memory_order_relaxed);
    do
    {
      __node->*_NextPtr = __old_head;
    } while (!__head_.compare_exchange_weak(__old_head, __node, _CUDA_VSTD::memory_order_acq_rel));

    if (__old_head != nullptr)
    {
      return false;
    }

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
    return __intrusive_queue<_NextPtr>::make_reversed(__head_.exchange(nullptr, _CUDA_VSTD::memory_order_acq_rel));
  }

private:
  __atomic_node_pointer __head_{nullptr};
};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_ATOMIC_INTRUSIVE_QUEUE
