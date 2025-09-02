// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ITERATOR_BACK_INSERT_ITERATOR_H
#define _CUDA_STD___ITERATOR_BACK_INSERT_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/iterator.h>
#include <cuda/std/__iterator/iterator.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Container>
class _CCCL_TYPE_VISIBILITY_DEFAULT __back_insert_iterator
{
protected:
  _Container* container;

public:
  using iterator_category = output_iterator_tag;
  using value_type        = void;
  using difference_type   = ptrdiff_t;
  using pointer           = void;
  using reference         = void;
  using container_type    = _Container;

  _CCCL_API constexpr explicit __back_insert_iterator(_Container& __x)
      : container(::cuda::std::addressof(__x))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __back_insert_iterator& operator=(const typename _Container::value_type& __value)
  {
    container->push_back(__value);
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __back_insert_iterator& operator=(typename _Container::value_type&& __value)
  {
    container->push_back(::cuda::std::move(__value));
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr __back_insert_iterator& operator*() noexcept
  {
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr const __back_insert_iterator& operator*() const noexcept
  {
    return *this;
  }

  _CCCL_API constexpr __back_insert_iterator& operator++() noexcept
  {
    return *this;
  }

  _CCCL_API constexpr __back_insert_iterator operator++(int) noexcept
  {
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr _Container* __get_container() const noexcept
  {
    return container;
  }
};

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Container>
class _CCCL_TYPE_VISIBILITY_DEFAULT back_insert_iterator
{
protected:
  _Container* container;

public:
  using iterator_category = output_iterator_tag;
  using value_type        = void;
#if _CCCL_STD_VER > 2017
  using difference_type = ptrdiff_t;
#else
  using difference_type = void;
#endif
  using pointer        = void;
  using reference      = void;
  using container_type = _Container;

  _CCCL_API constexpr explicit back_insert_iterator(_Container& __x)
      : container(::cuda::std::addressof(__x))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr back_insert_iterator& operator=(const typename _Container::value_type& __value)
  {
    container->push_back(__value);
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr back_insert_iterator& operator=(typename _Container::value_type&& __value)
  {
    container->push_back(::cuda::std::move(__value));
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr back_insert_iterator& operator*() noexcept
  {
    return *this;
  }

  _CCCL_API constexpr back_insert_iterator& operator++() noexcept
  {
    return *this;
  }

  _CCCL_API constexpr back_insert_iterator operator++(int) noexcept
  {
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr _Container* __get_container() const noexcept
  {
    return container;
  }
};
_CCCL_SUPPRESS_DEPRECATED_POP
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(back_insert_iterator);

template <class _Container>
[[nodiscard]] _CCCL_API constexpr back_insert_iterator<_Container> back_inserter(_Container& __x) noexcept
{
  return back_insert_iterator<_Container>(__x);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ITERATOR_BACK_INSERT_ITERATOR_H
