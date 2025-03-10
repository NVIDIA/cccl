// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_facade.h>

#include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN

//! \addtogroup iterators
//! \{

//! \addtogroup fancyiterator Fancy Iterators
//! \ingroup iterators
//! \{

//! \p offset_iterator wraps another iterator and an integral offset, apply the offset to the iterator when
//! dereferencing, comparing, or computing the distance between two offset_iterators. This is useful, when the
//! underlying iterator cannot be incremented, decremented, or advanced (e.g., because those operations are only
//! supported in device code).
//!
//! The following code snippet demonstrates how to create a \p offset_iterator:
//!
//! \code
//! #include <thrust/iterator/offset_iterator.h>
//! #include <thrust/fill.h>
//! #include <thrust/device_vector.h>
//!
//! int main()
//! {
//!   thrust::device_vector<int> data{1, 2, 3, 4};
//!   auto b = offset_iterator{data.begin(), 1};
//!   auto e = offset_iterator{data.end(), -1};
//!   thrust::fill(b, e, 42);
//!   // data is now [1, 42, 42, 4]
//!   ++b; // does not call ++ on the underlying iterator
//!   assert(b == e - 1);
//!
//!   return 0;
//! }
//! \endcode
template <typename Iterator, typename Offset = typename ::cuda::std::iterator_traits<Iterator>::difference_type>
class offset_iterator : public iterator_adaptor<offset_iterator<Iterator, Offset>, Iterator>
{
  //! \cond
  friend class iterator_core_access;
  using super_t = iterator_adaptor<offset_iterator<Iterator, Offset>, Iterator>;

public:
  using reference       = typename super_t::reference;
  using difference_type = typename super_t::difference_type;
  //! \endcond

  _CCCL_HOST_DEVICE offset_iterator(Iterator it = {}, Offset offset = {})
      : super_t(::cuda::std::move(it))
      , m_offset(offset)
  {}

  _CCCL_HOST_DEVICE const Offset& offset() const
  {
    return m_offset;
  }

  _CCCL_HOST_DEVICE Offset& offset()
  {
    return m_offset;
  }

  //! \cond

private:
  _CCCL_HOST_DEVICE reference dereference() const
  {
    return *(this->base() + static_cast<difference_type>(m_offset));
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE bool equal(const offset_iterator& other) const
  {
    return (this->base() + static_cast<difference_type>(m_offset))
        == (other.base() + static_cast<difference_type>(other.m_offset));
  }

  _CCCL_HOST_DEVICE void advance(difference_type n)
  {
    m_offset += n;
  }

  _CCCL_HOST_DEVICE void increment()
  {
    ++m_offset;
  }

  _CCCL_HOST_DEVICE void decrement()
  {
    --m_offset;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE difference_type distance_to(const offset_iterator& other) const
  {
    return (other.base() + static_cast<difference_type>(other.m_offset))
         - (this->base() + static_cast<difference_type>(m_offset));
  }

  Offset m_offset;
  //! \endcond
};

template <typename Iterator>
_CCCL_HOST_DEVICE offset_iterator(Iterator) -> offset_iterator<Iterator>;

//! \} // end fancyiterators
//! \} // end iterators

THRUST_NAMESPACE_END
