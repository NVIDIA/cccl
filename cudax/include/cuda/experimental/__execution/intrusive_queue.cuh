//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//                         Copyright (c) 2021-2022 Facebook, Inc & AFFILIATES.
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_INTRUSIVE_QUEUE
#define __CUDAX_EXECUTION_INTRUSIVE_QUEUE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__utility/exchange.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <auto _Next>
class _CCCL_TYPE_VISIBILITY_DEFAULT __intrusive_queue;

template <class _Item, _Item* _Item::* _Next>
class _CCCL_TYPE_VISIBILITY_DEFAULT __intrusive_queue<_Next>
{
public:
  _CCCL_HIDE_FROM_ABI __intrusive_queue() noexcept = default;

  _CCCL_API __intrusive_queue(__intrusive_queue&& __other) noexcept
      : __head_(::cuda::std::exchange(__other.__head_, nullptr))
      , __tail_(::cuda::std::exchange(__other.__tail_, nullptr))
  {}

  _CCCL_API auto operator=(__intrusive_queue&& __other) noexcept -> __intrusive_queue&
  {
    __head_ = ::cuda::std::exchange(__other.__head_, nullptr);
    __tail_ = ::cuda::std::exchange(__other.__tail_, nullptr);
    return *this;
  }

  _CCCL_API ~__intrusive_queue()
  {
    _CCCL_ASSERT(empty(), "");
  }

  [[nodiscard]]
  _CCCL_API static auto make_reversed(_Item* __list) noexcept -> __intrusive_queue
  {
    _Item* __new_head = nullptr;
    _Item* __new_tail = __list;

    while (__list != nullptr)
    {
      _Item* __next  = __list->*_Next;
      __list->*_Next = __new_head;
      __new_head     = __list;
      __list         = __next;
    }

    __intrusive_queue __result;
    __result.__head_ = __new_head;
    __result.__tail_ = __new_tail;
    return __result;
  }

  [[nodiscard]]
  _CCCL_API static auto make(_Item* __list) noexcept -> __intrusive_queue
  {
    __intrusive_queue __result{};
    __result.__head_ = __list;
    __result.__tail_ = __list;
    if (__list == nullptr)
    {
      return __result;
    }
    while (__result.__tail_->*_Next != nullptr)
    {
      __result.__tail_ = __result.__tail_->*_Next;
    }
    return __result;
  }

  [[nodiscard]]
  _CCCL_API auto empty() const noexcept -> bool
  {
    return __head_ == nullptr;
  }

  _CCCL_API void clear() noexcept
  {
    __head_ = nullptr;
    __tail_ = nullptr;
  }

  [[nodiscard]]
  _CCCL_API auto pop_front() noexcept -> _Item*
  {
    _CCCL_ASSERT(!empty(), "");
    _Item* __item = ::cuda::std::exchange(__head_, __head_->*_Next);
    // This should test if __head_ == nullptr, but due to a bug in
    // nvc++'s optimization, `__head_` isn't assigned until later.
    // Filed as NVBug#3952534.
    if (__item->*_Next == nullptr)
    {
      __tail_ = nullptr;
    }
    return __item;
  }

  _CCCL_API void push_front(_Item* __item) noexcept
  {
    _CCCL_ASSERT(__item != nullptr, "");
    __item->*_Next = __head_;
    __head_        = __item;
    if (__tail_ == nullptr)
    {
      __tail_ = __item;
    }
  }

  _CCCL_API void push_back(_Item* __item) noexcept
  {
    _CCCL_ASSERT(__item != nullptr, "");
    __item->*_Next                        = nullptr;
    (empty() ? __head_ : __tail_->*_Next) = __item;
    __tail_                               = __item;
  }

  _CCCL_API void append(__intrusive_queue __other) noexcept
  {
    if (!__other.empty())
    {
      (empty() ? __head_ : __tail_->*_Next) = ::cuda::std::exchange(__other.__head_, nullptr);
      __tail_                               = ::cuda::std::exchange(__other.__tail_, nullptr);
    }
  }

  _CCCL_API void prepend(__intrusive_queue __other) noexcept
  {
    if (!__other.empty())
    {
      __other.__tail_->*_Next = __head_;
      __head_                 = __other.__head_;
      if (__tail_ == nullptr)
      {
        __tail_ = __other.__tail_;
      }

      __other.clear();
    }
  }

  struct _CCCL_TYPE_VISIBILITY_DEFAULT iterator
  {
    using value_type _CCCL_NODEBUG_ALIAS        = _Item*;
    using difference_type _CCCL_NODEBUG_ALIAS   = ::cuda::std::ptrdiff_t;
    using pointer _CCCL_NODEBUG_ALIAS           = _Item* const*;
    using reference _CCCL_NODEBUG_ALIAS         = _Item* const&;
    using iterator_category _CCCL_NODEBUG_ALIAS = ::cuda::std::forward_iterator_tag;

    _CCCL_HIDE_FROM_ABI iterator() noexcept = default;

    _CCCL_API explicit iterator(_Item* __pred, _Item* __item) noexcept
        : __predecessor_(__pred)
        , __item_(__item)
    {}

    [[nodiscard]]
    _CCCL_API auto operator*() const noexcept -> _Item* const&
    {
      _CCCL_ASSERT(__item_ != nullptr, "");
      return __item_;
    }

    [[nodiscard]]
    _CCCL_API auto operator->() const noexcept -> _Item* const*
    {
      _CCCL_ASSERT(__item_ != nullptr, "");
      return &__item_;
    }

    _CCCL_API auto operator++() noexcept -> iterator&
    {
      _CCCL_ASSERT(__item_ != nullptr, "");
      __predecessor_ = ::cuda::std::exchange(__item_, __item_->*_Next);
      return *this;
    }

    _CCCL_API auto operator++(int) noexcept -> iterator
    {
      iterator __result = *this;
      ++*this;
      return __result;
    }

    [[nodiscard]]
    _CCCL_API friend auto operator==(const iterator& __lhs, const iterator& __rhs) noexcept -> bool
    {
      return __lhs.__item_ == __rhs.__item_;
    }

    [[nodiscard]]
    _CCCL_API friend auto operator!=(const iterator& __lhs, const iterator& __rhs) noexcept -> bool
    {
      return __lhs.__item_ != __rhs.__item_;
    }

    _Item* __predecessor_ = nullptr;
    _Item* __item_        = nullptr;
  };

  [[nodiscard]]
  _CCCL_API auto begin() const noexcept -> iterator
  {
    return iterator(nullptr, __head_);
  }

  [[nodiscard]]
  _CCCL_API auto end() const noexcept -> iterator
  {
    return iterator(__tail_, nullptr);
  }

  _CCCL_API void splice(iterator pos, __intrusive_queue& other, iterator first, iterator last) noexcept
  {
    if (first == last)
    {
      return;
    }
    _CCCL_ASSERT(first.__item_ != nullptr, "");
    _CCCL_ASSERT(last.__predecessor_ != nullptr, "");
    if (other.__head_ == first.__item_)
    {
      other.__head_ = last.__item_;
      if (other.__head_ == nullptr)
      {
        other.__tail_ = nullptr;
      }
    }
    else
    {
      _CCCL_ASSERT(first.__predecessor_ != nullptr, "");
      first.__predecessor_->*_Next = last.__item_;
      last.__predecessor_->*_Next  = pos.__item_;
    }
    if (empty())
    {
      __head_ = first.__item_;
      __tail_ = last.__predecessor_;
    }
    else
    {
      pos.__predecessor_->*_Next = first.__item_;
      if (pos.__item_ == nullptr)
      {
        __tail_ = last.__predecessor_;
      }
    }
  }

  _CCCL_API auto front() const noexcept -> _Item* const&
  {
    return __head_;
  }

  _CCCL_API auto back() const noexcept -> _Item* const&
  {
    return __tail_;
  }

private:
  _CCCL_API explicit __intrusive_queue(_Item* __head, _Item* __tail) noexcept
      : __head_(__head)
      , __tail_(__tail)
  {}

  _Item* __head_ = nullptr;
  _Item* __tail_ = nullptr;
};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_INTRUSIVE_QUEUE
