// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#include <thrust/iterator/iterator_adaptor.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

CUB_NAMESPACE_BEGIN

// Options for specifying memory aliasing
enum class MayAlias
{
  Yes,
  No
};

// Options for specifying sorting order.
enum class SortOrder
{
  Ascending,
  Descending
};

// Options for specifying the behavior of the stream compaction algorithm.
enum class SelectImpl
{
  // Stream compaction, discarding rejected items. It's required that memory of input and output are disjoint.
  Select,
  // Stream compaction, discarding rejected items. Memory of the input may be identical to the memory of the output.
  SelectPotentiallyInPlace,
  // Partition, keeping rejected items. It's required that memory of input and output are disjoint.
  Partition
};

namespace detail
{

template <typename Iterator, typename OffsetItT>
class offset_iterator_t : public THRUST_NS_QUALIFIER::iterator_adaptor<offset_iterator_t<Iterator, OffsetItT>, Iterator>
{
public:
  using super_t = THRUST_NS_QUALIFIER::iterator_adaptor<offset_iterator_t<Iterator, OffsetItT>, Iterator>;

  offset_iterator_t() = default;

  _CCCL_HOST_DEVICE offset_iterator_t(const Iterator& it, OffsetItT offset_it)
      : super_t(it)
      , offset_it(offset_it)
  {}

  // befriend thrust::iterator_core_access to allow it access to the private interface below
  friend class THRUST_NS_QUALIFIER::iterator_core_access;

private:
  OffsetItT offset_it;

  _CCCL_HOST_DEVICE typename super_t::reference dereference() const
  {
    return *(this->base() + (*offset_it));
  }
};

template <typename Iterator, typename OffsetItT>
_CCCL_HOST_DEVICE offset_iterator_t<Iterator, OffsetItT> make_offset_iterator(const Iterator& it, OffsetItT offset_it)
{
  return offset_iterator_t<Iterator, OffsetItT>{it, offset_it};
}

} // namespace detail

CUB_NAMESPACE_END
