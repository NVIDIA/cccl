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

#include <thrust/iterator/detail/device_system_tag.h>
#include <thrust/iterator/detail/host_system_tag.h>
#include <thrust/iterator/iterator_categories.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
_CCCL_HOST_DEVICE auto cat_to_system_impl(...) -> void;

_CCCL_HOST_DEVICE auto cat_to_system_impl(const input_host_iterator_tag&) -> host_system_tag;
_CCCL_HOST_DEVICE auto cat_to_system_impl(const output_host_iterator_tag&) -> host_system_tag;
_CCCL_HOST_DEVICE auto cat_to_system_impl(const forward_host_iterator_tag&) -> host_system_tag;
_CCCL_HOST_DEVICE auto cat_to_system_impl(const bidirectional_host_iterator_tag&) -> host_system_tag;
_CCCL_HOST_DEVICE auto cat_to_system_impl(const random_access_host_iterator_tag&) -> host_system_tag;

_CCCL_HOST_DEVICE auto cat_to_system_impl(const input_device_iterator_tag&) -> device_system_tag;
_CCCL_HOST_DEVICE auto cat_to_system_impl(const output_device_iterator_tag&) -> device_system_tag;
_CCCL_HOST_DEVICE auto cat_to_system_impl(const forward_device_iterator_tag&) -> device_system_tag;
_CCCL_HOST_DEVICE auto cat_to_system_impl(const bidirectional_device_iterator_tag&) -> device_system_tag;
_CCCL_HOST_DEVICE auto cat_to_system_impl(const random_access_device_iterator_tag&) -> device_system_tag;

template <typename Category>
struct iterator_category_to_system
{
  using type = decltype(cat_to_system_impl(Category{}));
};

template <typename Category>
using iterator_category_to_system_t = decltype(cat_to_system_impl(Category{}));
} // namespace detail
THRUST_NAMESPACE_END
