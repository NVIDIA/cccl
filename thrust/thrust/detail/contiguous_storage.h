/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/iterator/detail/normal_iterator.h>

#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>

#include <stdexcept>

THRUST_NAMESPACE_BEGIN

namespace detail
{
struct copy_allocator_t
{};

struct allocator_mismatch_on_swap : std::runtime_error
{
  allocator_mismatch_on_swap()
      : std::runtime_error("swap called on containers with allocators that propagate on swap, but compare non-equal")
  {}
};

// XXX parameter T is redundant with parameter Alloc
template <typename T, typename Alloc>
class contiguous_storage
{
private:
  using alloc_traits = thrust::detail::allocator_traits<Alloc>;

public:
  using allocator_type  = Alloc;
  using value_type      = T;
  using pointer         = typename alloc_traits::pointer;
  using const_pointer   = typename alloc_traits::const_pointer;
  using size_type       = typename alloc_traits::size_type;
  using difference_type = typename alloc_traits::difference_type;
  using reference       = typename alloc_traits::reference;
  using const_reference = typename alloc_traits::const_reference;

  using iterator       = thrust::detail::normal_iterator<pointer>;
  using const_iterator = thrust::detail::normal_iterator<const_pointer>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE explicit contiguous_storage(const allocator_type& alloc = allocator_type());

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE explicit contiguous_storage(size_type n, const allocator_type& alloc = allocator_type());

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE explicit contiguous_storage(copy_allocator_t, const contiguous_storage& other);

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE explicit contiguous_storage(copy_allocator_t, const contiguous_storage& other, size_type n);

  contiguous_storage& operator=(const contiguous_storage& x) = delete;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE ~contiguous_storage();

  _CCCL_HOST_DEVICE size_type size() const;

  _CCCL_HOST_DEVICE size_type max_size() const;

  _CCCL_HOST_DEVICE pointer data();

  _CCCL_HOST_DEVICE const_pointer data() const;

  _CCCL_HOST_DEVICE iterator begin();

  _CCCL_HOST_DEVICE const_iterator begin() const;

  _CCCL_HOST_DEVICE iterator end();

  _CCCL_HOST_DEVICE const_iterator end() const;

  _CCCL_HOST_DEVICE reference operator[](size_type n);

  _CCCL_HOST_DEVICE const_reference operator[](size_type n) const;

  _CCCL_HOST_DEVICE allocator_type get_allocator() const;

  // note that allocate does *not* automatically call deallocate
  _CCCL_HOST_DEVICE void allocate(size_type n);

  _CCCL_HOST_DEVICE void deallocate() noexcept;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void swap(contiguous_storage& other)
  {
    using ::cuda::std::swap;
    swap(m_begin, other.m_begin);
    swap(m_size, other.m_size);

    // From C++ standard [container.reqmts]
    //   If allocator_traits<allocator_type>::propagate_on_container_swap::value is true, then allocator_type
    //   shall meet the Cpp17Swappable requirements and the allocators of a and b shall also be exchanged by calling
    //   swap as described in [swappable.requirements]. Otherwise, the allocators shall not be swapped, and the behavior
    //   is undefined unless a.get_allocator() == b.get_allocator().
    if constexpr (allocator_traits<Alloc>::propagate_on_container_swap::value)
    {
      swap(m_allocator, other.m_allocator);
    }
    else if constexpr (!allocator_traits<Alloc>::is_always_equal::value)
    {
      NV_IF_TARGET(NV_IS_DEVICE, (assert(m_allocator == other.m_allocator);), (if (m_allocator != other.m_allocator) {
                     throw allocator_mismatch_on_swap();
                   }));
    }
  }

  _CCCL_HOST_DEVICE void value_initialize_n(iterator first, size_type n);

  _CCCL_HOST_DEVICE void uninitialized_fill_n(iterator first, size_type n, const value_type& value);

  template <typename InputIterator>
  _CCCL_HOST_DEVICE iterator uninitialized_copy(InputIterator first, InputIterator last, iterator result);

  template <typename System, typename InputIterator>
  _CCCL_HOST_DEVICE iterator uninitialized_copy(
    thrust::execution_policy<System>& from_system, InputIterator first, InputIterator last, iterator result);

  template <typename InputIterator, typename Size>
  _CCCL_HOST_DEVICE iterator uninitialized_copy_n(InputIterator first, Size n, iterator result);

  template <typename System, typename InputIterator, typename Size>
  _CCCL_HOST_DEVICE iterator
  uninitialized_copy_n(thrust::execution_policy<System>& from_system, InputIterator first, Size n, iterator result);

  _CCCL_HOST_DEVICE void destroy(iterator first, iterator last) noexcept;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void deallocate_on_allocator_mismatch(const contiguous_storage& other) noexcept
  {
    if constexpr (allocator_traits<Alloc>::propagate_on_container_copy_assignment::value)
    {
      if (m_allocator != other.m_allocator)
      {
        deallocate();
      }
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void destroy_on_allocator_mismatch(
    const contiguous_storage& other, [[maybe_unused]] iterator first, [[maybe_unused]] iterator last) noexcept
  {
    if constexpr (allocator_traits<Alloc>::propagate_on_container_copy_assignment::value)
    {
      if (m_allocator != other.m_allocator)
      {
        destroy(first, last);
      }
    }
  }

  _CCCL_HOST_DEVICE void set_allocator(const allocator_type& alloc);

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void propagate_allocator(const contiguous_storage& other)
  {
    if constexpr (allocator_traits<Alloc>::propagate_on_container_copy_assignment::value)
    {
      m_allocator = other.m_allocator;
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void propagate_allocator(contiguous_storage& other)
  {
    if constexpr (allocator_traits<Alloc>::propagate_on_container_move_assignment::value)
    {
      m_allocator = ::cuda::std::move(other.m_allocator);
    }
  }

  // allow move assignment for a sane implementation of allocator propagation
  _CCCL_HOST_DEVICE contiguous_storage& operator=(contiguous_storage&& other);

  _CCCL_SYNTHESIZE_SEQUENCE_ACCESS(contiguous_storage, const_iterator)

private:
  // XXX we could inherit from this to take advantage of empty base class optimization
  allocator_type m_allocator;

  iterator m_begin;

  size_type m_size;

  friend _CCCL_HOST_DEVICE void swap(contiguous_storage& lhs, contiguous_storage& rhs) noexcept(noexcept(lhs.swap(rhs)))
  {
    lhs.swap(rhs);
  }
}; // end contiguous_storage
} // namespace detail

THRUST_NAMESPACE_END

#include <thrust/detail/contiguous_storage.inl>
