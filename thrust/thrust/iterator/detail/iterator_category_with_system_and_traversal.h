// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

THRUST_NAMESPACE_BEGIN
namespace detail
{
template <typename Category, typename System, typename Traversal>
struct iterator_category_with_system_and_traversal : Category
{};

// specialize iterator_category_to_system for iterator_category_with_system_and_traversal
template <typename Category>
struct iterator_category_to_system;

template <typename Category, typename System, typename Traversal>
struct iterator_category_to_system<iterator_category_with_system_and_traversal<Category, System, Traversal>>
{
  using type = System;
};

// specialize iterator_category_to_traversal for iterator_category_with_system_and_traversal
template <typename Category>
struct iterator_category_to_traversal;

template <typename Category, typename System, typename Traversal>
struct iterator_category_to_traversal<iterator_category_with_system_and_traversal<Category, System, Traversal>>
{
  using type = Traversal;
};
} // namespace detail
THRUST_NAMESPACE_END
