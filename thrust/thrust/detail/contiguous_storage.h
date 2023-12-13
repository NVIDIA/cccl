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

#include <thrust/iterator/detail/normal_iterator.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/detail/allocator/allocator_traits.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

struct copy_allocator_t {};

// XXX parameter T is redundant with parameter Alloc
template<typename T, typename Alloc>
  class contiguous_storage
{
  private:
    typedef thrust::detail::allocator_traits<Alloc> alloc_traits;

  public:
    typedef Alloc                                      allocator_type;
    typedef T                                          value_type;
    typedef typename alloc_traits::pointer             pointer;
    typedef typename alloc_traits::const_pointer       const_pointer;
    typedef typename alloc_traits::size_type           size_type;
    typedef typename alloc_traits::difference_type     difference_type;
    typedef typename alloc_traits::reference           reference;
    typedef typename alloc_traits::const_reference     const_reference;

    typedef thrust::detail::normal_iterator<pointer>       iterator;
    typedef thrust::detail::normal_iterator<const_pointer> const_iterator;

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_HOST_DEVICE
    explicit contiguous_storage(const allocator_type &alloc = allocator_type());

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_HOST_DEVICE
    explicit contiguous_storage(size_type n, const allocator_type &alloc = allocator_type());

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_HOST_DEVICE
    explicit contiguous_storage(copy_allocator_t, const contiguous_storage &other);

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_HOST_DEVICE
    explicit contiguous_storage(copy_allocator_t, const contiguous_storage &other, size_type n);

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_HOST_DEVICE
    ~contiguous_storage();

    _CCCL_HOST_DEVICE
    size_type size() const;

    _CCCL_HOST_DEVICE
    size_type max_size() const;

    _CCCL_HOST_DEVICE
    pointer data();

    _CCCL_HOST_DEVICE
    const_pointer data() const;

    _CCCL_HOST_DEVICE
    iterator begin();

    _CCCL_HOST_DEVICE
    const_iterator begin() const;

    _CCCL_HOST_DEVICE
    iterator end();

    _CCCL_HOST_DEVICE
    const_iterator end() const;

    _CCCL_HOST_DEVICE
    reference operator[](size_type n);

    _CCCL_HOST_DEVICE
    const_reference operator[](size_type n) const;

    _CCCL_HOST_DEVICE
    allocator_type get_allocator() const;

    // note that allocate does *not* automatically call deallocate
    _CCCL_HOST_DEVICE
    void allocate(size_type n);

    _CCCL_HOST_DEVICE
    void deallocate();

    _CCCL_HOST_DEVICE
    void swap(contiguous_storage &x);

    _CCCL_HOST_DEVICE
    void default_construct_n(iterator first, size_type n);

    _CCCL_HOST_DEVICE
    void uninitialized_fill_n(iterator first, size_type n, const value_type &value);

    template<typename InputIterator>
    _CCCL_HOST_DEVICE
    iterator uninitialized_copy(InputIterator first, InputIterator last, iterator result);

    template<typename System, typename InputIterator>
    _CCCL_HOST_DEVICE
    iterator uninitialized_copy(thrust::execution_policy<System> &from_system,
                                InputIterator first,
                                InputIterator last,
                                iterator result);

    template<typename InputIterator, typename Size>
    _CCCL_HOST_DEVICE
    iterator uninitialized_copy_n(InputIterator first, Size n, iterator result);

    template<typename System, typename InputIterator, typename Size>
    _CCCL_HOST_DEVICE
    iterator uninitialized_copy_n(thrust::execution_policy<System> &from_system,
                                  InputIterator first,
                                  Size n,
                                  iterator result);

    _CCCL_HOST_DEVICE
    void destroy(iterator first, iterator last);

    _CCCL_HOST_DEVICE
    void deallocate_on_allocator_mismatch(const contiguous_storage &other);

    _CCCL_HOST_DEVICE
    void destroy_on_allocator_mismatch(const contiguous_storage &other,
        iterator first, iterator last);

    _CCCL_HOST_DEVICE
    void set_allocator(const allocator_type &alloc);

    _CCCL_HOST_DEVICE
    bool is_allocator_not_equal(const allocator_type &alloc) const;

    _CCCL_HOST_DEVICE
    bool is_allocator_not_equal(const contiguous_storage &other) const;

    _CCCL_HOST_DEVICE
    void propagate_allocator(const contiguous_storage &other);

#if _CCCL_STD_VER >= 2011
    _CCCL_HOST_DEVICE
    void propagate_allocator(contiguous_storage &other);

    // allow move assignment for a sane implementation of allocator propagation
    // on move assignment
    _CCCL_HOST_DEVICE
    contiguous_storage &operator=(contiguous_storage &&other);
#endif // _CCCL_STD_VER >= 2011

  private:
    // XXX we could inherit from this to take advantage of empty base class optimization
    allocator_type m_allocator;

    iterator m_begin;

    size_type m_size;

    // disallow assignment
    contiguous_storage &operator=(const contiguous_storage &x);

    _CCCL_HOST_DEVICE
    void swap_allocators(true_type, const allocator_type &);

    _CCCL_HOST_DEVICE
    void swap_allocators(false_type, allocator_type &);

    _CCCL_HOST_DEVICE
    bool is_allocator_not_equal_dispatch(true_type, const allocator_type &) const;

    _CCCL_HOST_DEVICE
    bool is_allocator_not_equal_dispatch(false_type, const allocator_type &) const;

    _CCCL_HOST_DEVICE
    void deallocate_on_allocator_mismatch_dispatch(true_type, const contiguous_storage &other);

    _CCCL_HOST_DEVICE
    void deallocate_on_allocator_mismatch_dispatch(false_type, const contiguous_storage &other);

    _CCCL_HOST_DEVICE
    void destroy_on_allocator_mismatch_dispatch(true_type, const contiguous_storage &other,
        iterator first, iterator last);

    _CCCL_HOST_DEVICE
    void destroy_on_allocator_mismatch_dispatch(false_type, const contiguous_storage &other,
        iterator first, iterator last);

    _CCCL_HOST_DEVICE
    void propagate_allocator_dispatch(true_type, const contiguous_storage &other);

    _CCCL_HOST_DEVICE
    void propagate_allocator_dispatch(false_type, const contiguous_storage &other);

#if _CCCL_STD_VER >= 2011
    _CCCL_HOST_DEVICE
    void propagate_allocator_dispatch(true_type, contiguous_storage &other);

    _CCCL_HOST_DEVICE
    void propagate_allocator_dispatch(false_type, contiguous_storage &other);
#endif // _CCCL_STD_VER >= 2011
}; // end contiguous_storage

} // end detail

template<typename T, typename Alloc>
_CCCL_HOST_DEVICE
void swap(detail::contiguous_storage<T,Alloc> &lhs, detail::contiguous_storage<T,Alloc> &rhs);

THRUST_NAMESPACE_END

#include <thrust/detail/contiguous_storage.inl>

