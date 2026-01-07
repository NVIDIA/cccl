//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CONTAINER_BUFFER_H
#define _CUDA___CONTAINER_BUFFER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  if _CCCL_CUDA_COMPILATION()
#    include <cub/device/device_transform.cuh>
#  endif // _CCCL_CUDA_COMPILATION()

#  include <cuda/__container/heterogeneous_iterator.h>
#  include <cuda/__container/uninitialized_async_buffer.h>
#  include <cuda/__launch/host_launch.h>
#  include <cuda/__memory_resource/any_resource.h>
#  include <cuda/__memory_resource/get_memory_resource.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__memory_resource/synchronous_resource_adapter.h>
#  include <cuda/__runtime/ensure_current_context.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/reverse_iterator.h>
#  include <cuda/std/__memory/uninitialized_algorithms.h>
#  include <cuda/std/__ranges/access.h>
#  include <cuda/std/__ranges/concepts.h>
#  include <cuda/std/__ranges/size.h>
#  include <cuda/std/__ranges/unwrap_end.h>
#  include <cuda/std/__type_traits/is_trivially_copyable.h>
#  include <cuda/std/__utility/forward.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/cstdint>
#  include <cuda/std/initializer_list>

#  include <cuda/std/__cccl/prologue.h>

//! @file The \c buffer class provides a container of contiguous memory
_CCCL_BEGIN_NAMESPACE_CUDA

// Once we add support from options taken from the env we can list them here in
// addition to using is_same_v
template <class _Env>
inline constexpr bool __buffer_compatible_env = ::cuda::std::is_same_v<_Env, ::cuda::std::execution::env<>>;

//! @rst
//! .. _cudax-containers-async-vector:
//!
//! buffer
//! -------------
//!
//! ``buffer`` is a container that provides resizable typed storage allocated
//! from a given :ref:`memory resource
//! <libcudacxx-extended-api-memory-resources-resource>`. It handles alignment,
//! release and growth of the allocation. The elements are initialized during
//! construction, which may require a kernel launch.
//!
//! In addition to being type-safe, ``buffer`` also takes a set of
//! :ref:`properties <libcudacxx-extended-api-memory-resources-properties>` to
//! ensure that e.g. execution space constraints are checked at compile time.
//! However, only stateless properties can be forwarded. To use a stateful
//! property, implement :ref:`get_property(const buffer&, Property)
//! <libcudacxx-extended-api-memory-resources-properties>`.
//!
//! @endrst
//! @tparam _Tp the type to be stored in the buffer
//! @tparam _Properties... The properties the allocated memory satisfies
template <class _Tp, class... _Properties>
class buffer
{
public:
  using value_type             = _Tp;
  using reference              = _Tp&;
  using const_reference        = const _Tp&;
  using pointer                = _Tp*;
  using const_pointer          = const _Tp*;
  using iterator               = ::cuda::heterogeneous_iterator<_Tp, _Properties...>;
  using const_iterator         = ::cuda::heterogeneous_iterator<const _Tp, _Properties...>;
  using reverse_iterator       = ::cuda::std::reverse_iterator<iterator>;
  using const_reverse_iterator = ::cuda::std::reverse_iterator<const_iterator>;
  using size_type              = ::cuda::std::size_t;
  using difference_type        = ::cuda::std::ptrdiff_t;
  using properties_list        = ::cuda::mr::properties_list<_Properties...>;

  using __buffer_t       = ::cuda::__uninitialized_async_buffer<_Tp, _Properties...>;
  using __resource_t     = ::cuda::mr::any_resource<_Properties...>;
  using __resource_ref_t = ::cuda::mr::resource_ref<_Properties...>;

  template <class, class...>
  friend class buffer;

  // For now we require trivially copyable type to simplify the implementation
  static_assert(::cuda::std::is_trivially_copyable_v<_Tp>, "cuda::buffer requires T to be trivially copyable.");

  // At least one of the properties must signal an execution space
  static_assert(::cuda::mr::__contains_execution_space_property<_Properties...>,
                "The properties of cuda::buffer must contain at "
                "least one execution space property!");

private:
  __buffer_t __buf_;

  //! @brief Helper to check container is compatible with this buffer
  template <class _Range>
  static constexpr bool __compatible_range = (::cuda::std::ranges::__container_compatible_range<_Range, _Tp>);

  //! @brief Helper to check whether a different buffer still satisfies all
  //! properties of this one
  template <class... _OtherProperties>
  static constexpr bool __properties_match =
    ::cuda::std::__type_set_contains_v<::cuda::std::__make_type_set<_OtherProperties...>, _Properties...>;

  //! @brief Helper to return an resource_ref to the currently used resource.
  //! Used to grow the buffer
  __resource_ref_t __borrow_resource() const noexcept
  {
    return const_cast<__resource_t&>(__buf_.memory_resource());
  }

  //! @brief Copies \p __count elements from `[__first, __last)` to \p __dest,
  //! where \p __first and \p __dest reside in the different memory spaces
  //! @param __first Pointer to the start of the input segment.
  //! @param __last Pointer to the end of the input segment.
  //! @param __dest Pointer to the start of the output segment.
  //! @param __count The number of elements to be copied.
  //! @note This function is inherently asynchronous. We need to ensure that the
  //! memory pointed to by \p __first and
  //! \p __last lives long enough
  template <class _Iter>
  _CCCL_HIDE_FROM_ABI void __copy_cross(_Iter __first, [[maybe_unused]] _Iter __last, pointer __dest, size_type __count)
  {
    if (__count == 0)
    {
      return;
    }

    static_assert(::cuda::std::contiguous_iterator<_Iter>, "Non contiguous iterators are not supported");
    // TODO use batched memcpy for non-contiguous iterators, it allows to
    // specify stream ordered access
    ::cuda::__driver::__memcpyAsync(
      __dest, ::cuda::std::to_address(__first), sizeof(_Tp) * __count, __buf_.stream().get());
  }

public:
  //! @addtogroup construction
  //! @{

  //! @brief Copy-constructs from a buffer
  //! @param __other The other buffer.
  _CCCL_HIDE_FROM_ABI explicit buffer(const buffer& __other)
      : __buf_(__other.memory_resource(), __other.stream(), __other.size())
  {
    this->__copy_cross<const_pointer>(
      __other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin(), __other.size());
  }

  //! @brief Move-constructs from a buffer
  //! @param __other The other buffer. After move construction, the other buffer
  //! can only be assigned to or destroyed.
  _CCCL_HIDE_FROM_ABI buffer(buffer&& __other) noexcept
      : __buf_(::cuda::std::move(__other.__buf_))
  {}

  //! @brief Copy-constructs from a buffer with matching properties
  //! @param __other The other buffer.
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HIDE_FROM_ABI explicit buffer(const buffer<_Tp, _OtherProperties...>& __other)
      : __buf_(__other.memory_resource(), __other.stream(), __other.size())
  {
    this->__copy_cross<const_pointer>(
      __other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin(), __other.size());
  }

  //! @brief Move-constructs from a buffer with matching properties
  //! @param __other The other buffer. After move construction, the other buffer
  //! can only be assigned to or destroyed.
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HOST_API buffer(buffer<_Tp, _OtherProperties...>&& __other) noexcept
      : __buf_(::cuda::std::move(__other.__buf_))
  {}

  //! @brief Constructs an empty buffer using an environment
  //! @param __env The environment providing the needed information
  //! @note No memory is allocated.
  _CCCL_TEMPLATE(class _Resource, class _Env = ::cuda::std::execution::env<>)
  _CCCL_REQUIRES(
    ::cuda::mr::synchronous_resource<::cuda::std::decay_t<_Resource>> _CCCL_AND __buffer_compatible_env<_Env>)
  _CCCL_HIDE_FROM_ABI
  buffer(::cuda::stream_ref __stream, _Resource&& __resource, [[maybe_unused]] const _Env& __env = {})
      : __buf_(::cuda::mr::__adapt_if_synchronous(::cuda::std::forward<_Resource>(__resource)), __stream, 0)
  {
    static_assert(::cuda::std::is_copy_constructible_v<::cuda::std::decay_t<_Resource>>,
                  "Buffer owns a copy of the memory resource, which means it must be copy constructible. "
                  "cuda::mr::shared_resource can be used to attach shared ownership to a resource type.");
  }

  //! @brief Constructs a buffer of size \p __size using a memory and leaves all
  //! elements uninitialized
  //! @param __env The environment used to query the memory resource.
  //! @param __size The size of the buffer.
  //! @warning This constructor does *NOT* initialize any elements. It is the
  //! user's responsibility to ensure that the elements within `[vec.begin(),
  //! vec.end())` are properly initialized, e.g with
  //! `cuda::std::uninitialized_copy`. At the destruction of the \c buffer all
  //! elements in the range `[vec.begin(), vec.end())` will be destroyed.
  _CCCL_TEMPLATE(class _Resource, class _Env = ::cuda::std::execution::env<>)
  _CCCL_REQUIRES(
    ::cuda::mr::synchronous_resource<::cuda::std::decay_t<_Resource>> _CCCL_AND __buffer_compatible_env<_Env>)
  _CCCL_HIDE_FROM_ABI explicit buffer(
    ::cuda::stream_ref __stream,
    _Resource&& __resource,
    const size_type __size,
    ::cuda::no_init_t,
    [[maybe_unused]] const _Env& __env = {})
      : __buf_(::cuda::mr::__adapt_if_synchronous(::cuda::std::forward<_Resource>(__resource)), __stream, __size)
  {
    static_assert(::cuda::std::is_copy_constructible_v<::cuda::std::decay_t<_Resource>>,
                  "Buffer owns a copy of the memory resource, which means it must be copy constructible. "
                  "cuda::mr::shared_resource can be used to attach shared ownership to a resource type.");
  }

  //! @brief Constructs a buffer using a memory resource and copy-constructs all
  //! elements from the forward range
  //! ``[__first, __last)``
  //! @param __env The environment used to query the memory resource.
  //! @param __first The start of the input sequence.
  //! @param __last The end of the input sequence.
  //! @note If `__first == __last` then no memory is allocated
  _CCCL_TEMPLATE(class _Iter, class _Resource, class _Env = ::cuda::std::execution::env<>)
  _CCCL_REQUIRES(::cuda::mr::synchronous_resource<::cuda::std::decay_t<_Resource>>
                   _CCCL_AND ::cuda::std::__has_forward_traversal<_Iter>)
  _CCCL_HIDE_FROM_ABI
  buffer(::cuda::stream_ref __stream,
         _Resource&& __resource,
         _Iter __first,
         _Iter __last,
         [[maybe_unused]] const _Env& __env = {})
      : __buf_(::cuda::mr::__adapt_if_synchronous(::cuda::std::forward<_Resource>(__resource)),
               __stream,
               static_cast<size_type>(::cuda::std::distance(__first, __last)))
  {
    static_assert(::cuda::std::is_copy_constructible_v<::cuda::std::decay_t<_Resource>>,
                  "Buffer owns a copy of the memory resource, which means it must be copy constructible. "
                  "cuda::mr::shared_resource can be used to attach shared ownership to a resource type.");
    this->__copy_cross<_Iter>(__first, __last, __unwrapped_begin(), __buf_.size());
  }

  //! @brief Constructs a buffer using a memory resource and copy-constructs all
  //! elements from \p __ilist
  //! @param __env The environment used to query the memory resource.
  //! @param __ilist The initializer_list being copied into the buffer.
  //! @note If `__ilist.size() == 0` then no memory is allocated
  _CCCL_TEMPLATE(class _Resource, class _Env = ::cuda::std::execution::env<>)
  _CCCL_REQUIRES(
    ::cuda::mr::synchronous_resource<::cuda::std::decay_t<_Resource>> _CCCL_AND __buffer_compatible_env<_Env>)
  _CCCL_HIDE_FROM_ABI
  buffer(::cuda::stream_ref __stream,
         _Resource&& __resource,
         ::cuda::std::initializer_list<_Tp> __ilist,
         [[maybe_unused]] const _Env& __env = {})
      : __buf_(::cuda::mr::__adapt_if_synchronous(::cuda::std::forward<_Resource>(__resource)), __stream, __ilist.size())
  {
    static_assert(::cuda::std::is_copy_constructible_v<::cuda::std::decay_t<_Resource>>,
                  "Buffer owns a copy of the memory resource, which means it must be copy constructible. "
                  "cuda::mr::shared_resource can be used to attach shared ownership to a resource type.");
    this->__copy_cross(__ilist.begin(), __ilist.end(), __unwrapped_begin(), __buf_.size());
  }

  //! @brief Constructs a buffer using a memory resource and an input range
  //! @param __env The environment used to query the memory resource.
  //! @param __range The input range to be moved into the buffer.
  //! @note If `__range.size() == 0` then no memory is allocated.
  _CCCL_TEMPLATE(class _Range, class _Resource, class _Env = ::cuda::std::execution::env<>)
  _CCCL_REQUIRES(
    ::cuda::mr::synchronous_resource<::cuda::std::decay_t<_Resource>> _CCCL_AND __compatible_range<_Range>
      _CCCL_AND ::cuda::std::ranges::forward_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>)
  _CCCL_HIDE_FROM_ABI
  buffer(::cuda::stream_ref __stream, _Resource&& __resource, _Range&& __range, [[maybe_unused]] const _Env& __env = {})
      : __buf_(::cuda::mr::__adapt_if_synchronous(::cuda::std::forward<_Resource>(__resource)),
               __stream,
               static_cast<size_type>(::cuda::std::ranges::size(__range)))
  {
    static_assert(::cuda::std::is_copy_constructible_v<::cuda::std::decay_t<_Resource>>,
                  "Buffer owns a copy of the memory resource, which means it must be copy constructible. "
                  "cuda::mr::shared_resource can be used to attach shared ownership to a resource type.");
    using _Iter = ::cuda::std::ranges::iterator_t<_Range>;
    this->__copy_cross<_Iter>(
      ::cuda::std::ranges::begin(__range),
      ::cuda::std::ranges::__unwrap_end(__range),
      __unwrapped_begin(),
      __buf_.size());
  }

#  ifndef _CCCL_DOXYGEN_INVOKED // doxygen conflates the overloads
  _CCCL_TEMPLATE(class _Range, class _Resource, class _Env = ::cuda::std::execution::env<>)
  _CCCL_REQUIRES(
    ::cuda::mr::synchronous_resource<::cuda::std::decay_t<_Resource>> _CCCL_AND __compatible_range<_Range>
      _CCCL_AND ::cuda::std::ranges::forward_range<_Range> _CCCL_AND(!::cuda::std::ranges::sized_range<_Range>))
  _CCCL_HIDE_FROM_ABI
  buffer(::cuda::stream_ref __stream, _Resource&& __resource, _Range&& __range, [[maybe_unused]] const _Env& __env = {})
      : __buf_(::cuda::mr::__adapt_if_synchronous(::cuda::std::forward<_Resource>(__resource)),
               __stream,
               static_cast<size_type>(
                 ::cuda::std::ranges::distance(::cuda::std::ranges::begin(__range), ::cuda::std::ranges::end(__range))),
               __env)
  {
    static_assert(::cuda::std::is_copy_constructible_v<::cuda::std::decay_t<_Resource>>,
                  "Buffer owns a copy of the memory resource, which means it must be copy constructible. "
                  "cuda::mr::shared_resource can be used to attach shared ownership to a resource type.");
    using _Iter = ::cuda::std::ranges::iterator_t<_Range>;
    this->__copy_cross<_Iter>(
      ::cuda::std::ranges::begin(__range),
      ::cuda::std::ranges::__unwrap_end(__range),
      __unwrapped_begin(),
      __buf_.size());
  }
#  endif // _CCCL_DOXYGEN_INVOKED
  //! @}

  //! @addtogroup iterators
  //! @{
  //! @brief Returns an iterator to the first element of the buffer. If the
  //! buffer is empty, the returned iterator will be equal to end().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI iterator begin() noexcept
  {
    return iterator{__buf_.data()};
  }

  //! @brief Returns an immutable iterator to the first element of the buffer.
  //! If the buffer is empty, the returned iterator will be equal to end().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_iterator begin() const noexcept
  {
    return const_iterator{__buf_.data()};
  }

  //! @brief Returns an immutable iterator to the first element of the buffer.
  //! If the buffer is empty, the returned iterator will be equal to end().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_iterator cbegin() const noexcept
  {
    return const_iterator{__buf_.data()};
  }

  //! @brief Returns an iterator to the element following the last element of
  //! the buffer. This element acts as a placeholder; attempting to access it
  //! results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI iterator end() noexcept
  {
    return iterator{__buf_.data() + __buf_.size()};
  }

  //! @brief Returns an immutable iterator to the element following the last
  //! element of the buffer. This element acts as a placeholder; attempting to
  //! access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_iterator end() const noexcept
  {
    return const_iterator{__buf_.data() + __buf_.size()};
  }

  //! @brief Returns an immutable iterator to the element following the last
  //! element of the buffer. This element acts as a placeholder; attempting to
  //! access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_iterator cend() const noexcept
  {
    return const_iterator{__buf_.data() + __buf_.size()};
  }

  //! @brief Returns a reverse iterator to the first element of the reversed
  //! buffer. It corresponds to the last element of the non-reversed buffer. If
  //! the buffer is empty, the returned iterator is equal to rend().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI reverse_iterator rbegin() noexcept
  {
    return reverse_iterator{end()};
  }

  //! @brief Returns an immutable reverse iterator to the first element of the
  //! reversed buffer. It corresponds to the last element of the non-reversed
  //! buffer. If the buffer is empty, the returned iterator is equal to rend().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reverse_iterator rbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns an immutable reverse iterator to the first element of the
  //! reversed buffer. It corresponds to the last element of the non-reversed
  //! buffer. If the buffer is empty, the returned iterator is equal to rend().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reverse_iterator crbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns a reverse iterator to the element following the last
  //! element of the reversed buffer. It corresponds to the element preceding
  //! the first element of the non-reversed buffer. This element acts as a
  //! placeholder, attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI reverse_iterator rend() noexcept
  {
    return reverse_iterator{begin()};
  }

  //! @brief Returns an immutable reverse iterator to the element following the
  //! last element of the reversed buffer. It corresponds to the element
  //! preceding the first element of the non-reversed buffer. This element acts
  //! as a placeholder, attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reverse_iterator rend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns an immutable reverse iterator to the element following the
  //! last element of the reversed buffer. It corresponds to the element
  //! preceding the first element of the non-reversed buffer. This element acts
  //! as a placeholder, attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reverse_iterator crend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns a pointer to the first element of the buffer. If the buffer
  //! has not allocated memory the pointer will be null.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI pointer data() noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a pointer to the first element of the buffer. If the buffer
  //! has not allocated memory the pointer will be null.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_pointer data() const noexcept
  {
    return __buf_.data();
  }

#  ifndef _CCCL_DOXYGEN_INVOKED
  //! @brief Returns a pointer to the first element of the buffer. If the buffer
  //! is empty, the returned pointer will be null.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI pointer __unwrapped_begin() noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a const pointer to the first element of the buffer. If the
  //! buffer is empty, the returned pointer will be null.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_pointer __unwrapped_begin() const noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a pointer to the element following the last element of the
  //! buffer. This element acts as a placeholder; attempting to access it
  //! results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI pointer __unwrapped_end() noexcept
  {
    return __buf_.data() + __buf_.size();
  }

  //! @brief Returns a const pointer to the element following the last element
  //! of the buffer. This element acts as a placeholder; attempting to access it
  //! results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_pointer __unwrapped_end() const noexcept
  {
    return __buf_.data() + __buf_.size();
  }
#  endif // _CCCL_DOXYGEN_INVOKED

  //! @}

  //! @addtogroup access
  //! @brief Returns a reference to the \p __n 'th element of the async_vector
  //! @param __n The index of the element we want to access
  //! @note Does not synchronize with the stored stream
  [[nodiscard]] _CCCL_HIDE_FROM_ABI reference get_unsynchronized(const size_type __n) noexcept
  {
    _CCCL_ASSERT(__n < __buf_.size(), "cuda::buffer::get_unsynchronized out of range!");
    return __unwrapped_begin()[__n];
  }

  //! @brief Returns a reference to the \p __n 'th element of the async_vector
  //! @param __n The index of the element we want to access
  //! @note Does not synchronize with the stored stream
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reference get_unsynchronized(const size_type __n) const noexcept
  {
    _CCCL_ASSERT(__n < __buf_.size(), "cuda::buffer::get_unsynchronized out of range!");
    return __unwrapped_begin()[__n];
  }

  //! @}

  //! @addtogroup size
  //! @{
  //! @brief Returns the current number of elements stored in the buffer.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI size_type size() const noexcept
  {
    return __buf_.size();
  }

  //! @brief Returns true if the buffer is empty.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI bool empty() const noexcept
  {
    return __buf_.size() == 0;
  }
  //! @}

  //! @rst
  //! Returns a \c const reference to the :ref:`any_resource
  //! <cuda-memory-resource-any-resource>` that holds the memory resource used
  //! to allocate the buffer
  //! @endrst
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const __resource_t& memory_resource() const noexcept
  {
    return __buf_.memory_resource();
  }

  //! @brief Returns the stored stream
  //! @note Stream used to allocate the buffer is initially stored in the
  //! buffer, but can be changed with `set_stream`
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr stream_ref stream() const noexcept
  {
    return __buf_.stream();
  }

  //! @brief Replaces the stored stream
  //! @param __new_stream the new stream
  //! @note Always synchronizes with the old stream
  _CCCL_HIDE_FROM_ABI constexpr void set_stream(stream_ref __new_stream)
  {
    __buf_.set_stream_unsynchronized(__new_stream);
  }

  //! @brief Move assignment operator
  //! @param __other The other buffer. After move assignment, the other buffer
  //! can only be assigned to or destroyed.
  _CCCL_HIDE_FROM_ABI void operator=(buffer&& __other)
  {
    __buf_ = ::cuda::std::move(__other.__buf_);
  }

  //! @brief Swaps the contents of a buffer with those of \p __other
  //! @param __other The other buffer.
  _CCCL_HIDE_FROM_ABI void swap(buffer& __other) noexcept
  {
    ::cuda::std::swap(__buf_, __other.__buf_);
  }

  //! @brief Swaps the contents of two buffers
  //! @param __lhs One buffer.
  //! @param __rhs The other buffer.
  _CCCL_HIDE_FROM_ABI friend void swap(buffer& __lhs, buffer& __rhs) noexcept
  {
    __lhs.swap(__rhs);
  }

  //! @brief Destroys the buffer, deallocates the buffer and destroys the memory
  //! resource
  //! @param __stream The stream to deallocate the buffer on.
  //! @warning After this explicit destroy call, the buffer can only be assigned
  //! to or destroyed.
  _CCCL_HIDE_FROM_ABI void destroy(::cuda::stream_ref __stream)
  {
    __buf_.destroy(__stream);
  }

  //! @brief Destroys the buffer, deallocates the buffer and destroys the memory
  //! resource
  //! @note Uses the stored stream to deallocate the buffer, equivalent to
  //! calling buffer.destroy(buffer.stream())
  //! @warning After this explicit destroy call, the buffer can only be assigned
  //! to or destroyed.
  _CCCL_HIDE_FROM_ABI void destroy()
  {
    __buf_.destroy();
  }

  //! @brief Causes the buffer to be treated as a span when passed to
  //! cuda::launch.
  //! @pre The buffer must have the cuda::mr::device_accessible property.
  template <class _DeviceAccessible = ::cuda::mr::device_accessible>
  [[nodiscard]] _CCCL_HIDE_FROM_ABI friend auto transform_launch_argument(::cuda::stream_ref, buffer& __self) noexcept
    _CCCL_TRAILING_REQUIRES(::cuda::std::span<_Tp>)(::cuda::std::__is_included_in_v<_DeviceAccessible, _Properties...>)
  {
    return {__self.__unwrapped_begin(), __self.size()};
  }

  //! @brief Causes the buffer to be treated as a span when passed to
  //! cuda::launch
  //! @pre The buffer must have the cuda::mr::device_accessible property.
  template <class _DeviceAccessible = ::cuda::mr::device_accessible>
  [[nodiscard]] _CCCL_HIDE_FROM_ABI friend auto
  transform_launch_argument(::cuda::stream_ref, const buffer& __self) noexcept _CCCL_TRAILING_REQUIRES(
    ::cuda::std::span<const _Tp>)(::cuda::std::__is_included_in_v<_DeviceAccessible, _Properties...>)
  {
    return {__self.__unwrapped_begin(), __self.size()};
  }

  //! @brief Forwards the passed properties
  _CCCL_TEMPLATE(class _Property)
  _CCCL_REQUIRES((!property_with_value<_Property>) _CCCL_AND ::cuda::std::__is_included_in_v<_Property, _Properties...>)
  _CCCL_HIDE_FROM_ABI friend void get_property(const buffer&, _Property) noexcept {}
};

template <class _Tp>
using device_buffer = buffer<_Tp, ::cuda::mr::device_accessible>;

template <class _Tp>
using host_buffer = buffer<_Tp, ::cuda::mr::host_accessible>;

template <class _Tp, class _PropsList>
using __buffer_type_for_props = typename ::cuda::std::remove_reference_t<_PropsList>::template rebind<buffer, _Tp>;

template <typename _BufferTo, typename _BufferFrom>
void __copy_cross_buffers(stream_ref __stream, _BufferTo& __to, const _BufferFrom& __from)
{
  __stream.wait(__from.stream());
  ::cuda::__driver::__memcpyAsync(
    __to.__unwrapped_begin(),
    __from.__unwrapped_begin(),
    sizeof(typename _BufferTo::value_type) * __from.size(),
    __stream.get());
}

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

//! @brief Copy-constructs elements in the range `[__first, __first + __count)`.
//! @param __first Pointer to the first element to be initialized.
//! @param __count The number of elements to be initialized.
template <typename _Tp, mr::__memory_accessability _Accessability>
_CCCL_HIDE_FROM_ABI void
__fill_n(cuda::stream_ref __stream, _Tp* __first, ::cuda::std::size_t __count, const _Tp& __value)
{
  if (__count == 0)
  {
    return;
  }

  // We don't know what to do with both device and host accessible buffers, so
  // we need to check the attributes
  if constexpr (_Accessability == mr::__memory_accessability::__host_device)
  {
    __driver::__pointer_attribute_value_type_t<CU_POINTER_ATTRIBUTE_MEMORY_TYPE> __type;
    bool __is_managed{};
    auto __status1 = ::cuda::__driver::__pointerGetAttributeNoThrow<CU_POINTER_ATTRIBUTE_MEMORY_TYPE>(__type, __first);
    auto __status2 =
      ::cuda::__driver::__pointerGetAttributeNoThrow<CU_POINTER_ATTRIBUTE_IS_MANAGED>(__is_managed, __first);
    if (__status1 != ::cudaSuccess || __status2 != ::cudaSuccess)
    {
      __throw_cuda_error(__status1, "Failed to get buffer memory attributes");
    }
    if (__type == ::CU_MEMORYTYPE_HOST && !__is_managed)
    {
      __fill_n<_Tp, mr::__memory_accessability::__host>(__stream, __first, __count, __value);
    }
    else
    {
      __fill_n<_Tp, mr::__memory_accessability::__device>(__stream, __first, __count, __value);
    }
  }
  else if constexpr (_Accessability == mr::__memory_accessability::__host)
  {
    ::cuda::host_launch(
      __stream, ::cuda::std::uninitialized_fill_n<_Tp*, ::cuda::std::size_t, _Tp>, __first, __count, __value);
  }
  else
  {
    if constexpr (sizeof(_Tp) <= 4)
    {
      ::cuda::__driver::__memsetAsync(__first, __value, __count, __stream.get());
    }
    else
    {
#  if _CCCL_CUDA_COMPILATION()
      ::cuda::__ensure_current_context __guard(__stream);
      ::cub::DeviceTransform::Fill(__first, __count, __value, __stream.get());
#  else // ^^^ _CCCL_CUDA_COMPILATION() ^^^ / vvv !_CCCL_CUDA_COMPILATION() vvv
      static_assert(sizeof(_Tp) <= 4,
                    "CUDA compiler is required to initialize an async_buffer with elements larger than 4 bytes");
#  endif // ^^^ !_CCCL_CUDA_COMPILATION() ^^^
    }
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Tp,
               class _FirstProperty,
               class... _RestProperties,
               class _Resource,
               class... _SourceProperties,
               class _Env = ::cuda::std::execution::env<>)
_CCCL_REQUIRES(
  ::cuda::mr::synchronous_resource_with<::cuda::std::decay_t<_Resource>, _FirstProperty, _RestProperties...> _CCCL_AND
    __buffer_compatible_env<_Env>)
buffer<_Tp, _FirstProperty, _RestProperties...> make_buffer(
  stream_ref __stream, _Resource&& __mr, const buffer<_Tp, _SourceProperties...>& __source, const _Env& __env = {})
{
  buffer<_Tp, _FirstProperty, _RestProperties...> __res{
    __stream, ::cuda::std::forward<_Resource>(__mr), __source.size(), no_init, __env};

  __copy_cross_buffers(__stream, __res, __source);

  return __res;
}

_CCCL_TEMPLATE(class _Tp, class _Resource, class... _SourceProperties, class _Env = ::cuda::std::execution::env<>)
_CCCL_REQUIRES(::cuda::mr::synchronous_resource<::cuda::std::decay_t<_Resource>>
                 _CCCL_AND ::cuda::mr::__has_default_queries<::cuda::std::decay_t<_Resource>>)
auto make_buffer(
  stream_ref __stream, _Resource&& __mr, const buffer<_Tp, _SourceProperties...>& __source, const _Env& __env = {})
{
  using __buffer_type = __buffer_type_for_props<_Tp, typename ::cuda::std::decay_t<_Resource>::default_queries>;
  auto __res          = __buffer_type{__stream, ::cuda::std::forward<_Resource>(__mr), __source.size(), no_init, __env};

  __copy_cross_buffers(__stream, __res, __source);

  return __res;
}

// Empty buffer make function
_CCCL_TEMPLATE(
  class _Tp, class _FirstProperty, class... _RestProperties, class _Resource, class _Env = ::cuda::std::execution::env<>)
_CCCL_REQUIRES(
  ::cuda::mr::synchronous_resource_with<::cuda::std::decay_t<_Resource>, _FirstProperty, _RestProperties...> _CCCL_AND
    __buffer_compatible_env<_Env>)
buffer<_Tp, _FirstProperty, _RestProperties...>
make_buffer(stream_ref __stream, _Resource&& __mr, const _Env& __env = {})
{
  return buffer<_Tp, _FirstProperty, _RestProperties...>{__stream, ::cuda::std::forward<_Resource>(__mr), __env};
}

_CCCL_TEMPLATE(class _Tp, class _Resource, class _Env = ::cuda::std::execution::env<>)
_CCCL_REQUIRES(::cuda::mr::synchronous_resource<::cuda::std::decay_t<_Resource>>
                 _CCCL_AND ::cuda::mr::__has_default_queries<::cuda::std::decay_t<_Resource>> _CCCL_AND
                   __buffer_compatible_env<_Env>)
auto make_buffer(stream_ref __stream, _Resource&& __mr, const _Env& __env = {})
{
  using __buffer_type = __buffer_type_for_props<_Tp, typename ::std::decay_t<_Resource>::default_queries>;
  return __buffer_type{__stream, ::cuda::std::forward<_Resource>(__mr), __env};
}

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

// Size and value make function
_CCCL_TEMPLATE(
  class _Tp, class _FirstProperty, class... _RestProperties, class _Resource, class _Env = ::cuda::std::execution::env<>)
_CCCL_REQUIRES(
  ::cuda::mr::synchronous_resource_with<::cuda::std::decay_t<_Resource>, _FirstProperty, _RestProperties...> _CCCL_AND
    __buffer_compatible_env<_Env>)
buffer<_Tp, _FirstProperty, _RestProperties...> make_buffer(
  stream_ref __stream, _Resource&& __mr, size_t __size, const _Tp& __value, [[maybe_unused]] const _Env& __env = {})
{
  auto __res =
    buffer<_Tp, _FirstProperty, _RestProperties...>{__stream, ::cuda::std::forward<_Resource>(__mr), __size, no_init};
  __fill_n<_Tp, mr::__memory_accessability_from_properties<_FirstProperty, _RestProperties...>::value>(
    __stream, __res.__unwrapped_begin(), __size, __value);
  return __res;
}

_CCCL_TEMPLATE(class _Tp, class _Resource, class _Env = ::cuda::std::execution::env<>)
_CCCL_REQUIRES(::cuda::mr::synchronous_resource<::cuda::std::decay_t<_Resource>>
                 _CCCL_AND ::cuda::mr::__has_default_queries<::cuda::std::decay_t<_Resource>>)
auto make_buffer(
  stream_ref __stream, _Resource&& __mr, size_t __size, const _Tp& __value, [[maybe_unused]] const _Env& __env = {})
{
  using __default_queries = typename ::cuda::std::decay_t<_Resource>::default_queries;
  using __buffer_type     = __buffer_type_for_props<_Tp, __default_queries>;
  auto __res              = __buffer_type{__stream, ::cuda::std::forward<_Resource>(__mr), __size, no_init};
  __fill_n<_Tp, __default_queries::template rebind<mr::__memory_accessability_from_properties>::value>(
    __stream, __res.__unwrapped_begin(), __size, __value);
  return __res;
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

// Size with no initialization make function
_CCCL_TEMPLATE(
  class _Tp, class _FirstProperty, class... _RestProperties, class _Resource, class _Env = ::cuda::std::execution::env<>)
_CCCL_REQUIRES(
  ::cuda::mr::synchronous_resource_with<::cuda::std::decay_t<_Resource>, _FirstProperty, _RestProperties...> _CCCL_AND
    __buffer_compatible_env<_Env>)
buffer<_Tp, _FirstProperty, _RestProperties...>
make_buffer(stream_ref __stream, _Resource&& __mr, size_t __size, ::cuda::no_init_t, const _Env& __env = {})
{
  return buffer<_Tp, _FirstProperty, _RestProperties...>{
    __stream, ::cuda::std::forward<_Resource>(__mr), __size, ::cuda::no_init, __env};
}

_CCCL_TEMPLATE(class _Tp, class _Resource, class _Env = ::cuda::std::execution::env<>)
_CCCL_REQUIRES(::cuda::mr::synchronous_resource<::cuda::std::decay_t<_Resource>>
                 _CCCL_AND ::cuda::mr::__has_default_queries<_Resource>)
auto make_buffer(stream_ref __stream, _Resource&& __mr, size_t __size, ::cuda::no_init_t, const _Env& __env = {})
{
  using __buffer_type = __buffer_type_for_props<_Tp, typename ::cuda::std::decay_t<_Resource>::default_queries>;
  return __buffer_type{__stream, ::cuda::std::forward<_Resource>(__mr), __size, ::cuda::no_init, __env};
}

// Iterator range make function
_CCCL_TEMPLATE(class _Tp,
               class _FirstProperty,
               class... _RestProperties,
               class _Resource,
               class _Iter,
               class _Env = ::cuda::std::execution::env<>)
_CCCL_REQUIRES(
  ::cuda::mr::synchronous_resource_with<::cuda::std::decay_t<_Resource>, _FirstProperty, _RestProperties...> _CCCL_AND
    __buffer_compatible_env<_Env> _CCCL_AND ::cuda::std::__has_forward_traversal<_Iter>)
buffer<_Tp, _FirstProperty, _RestProperties...>
make_buffer(stream_ref __stream, _Resource&& __mr, _Iter __first, _Iter __last, const _Env& __env = {})
{
  return buffer<_Tp, _FirstProperty, _RestProperties...>{
    __stream, ::cuda::std::forward<_Resource>(__mr), __first, __last, __env};
}

_CCCL_TEMPLATE(class _Tp, class _Resource, class _Iter, class _Env = ::cuda::std::execution::env<>)
_CCCL_REQUIRES(
  ::cuda::mr::synchronous_resource<::cuda::std::decay_t<_Resource>>
    _CCCL_AND ::cuda::mr::__has_default_queries<_Resource> _CCCL_AND ::cuda::std::__has_forward_traversal<_Iter>)
auto make_buffer(stream_ref __stream, _Resource&& __mr, _Iter __first, _Iter __last, const _Env& __env = {})
{
  using __buffer_type = __buffer_type_for_props<_Tp, typename ::cuda::std::decay_t<_Resource>::default_queries>;
  return __buffer_type{__stream, ::cuda::std::forward<_Resource>(__mr), __first, __last, __env};
}

// Initializer list make function
_CCCL_TEMPLATE(
  class _Tp, class _FirstProperty, class... _RestProperties, class _Resource, class _Env = ::cuda::std::execution::env<>)
_CCCL_REQUIRES(
  ::cuda::mr::synchronous_resource_with<::cuda::std::decay_t<_Resource>, _FirstProperty, _RestProperties...> _CCCL_AND
    __buffer_compatible_env<_Env>)
buffer<_Tp, _FirstProperty, _RestProperties...>
make_buffer(stream_ref __stream, _Resource&& __mr, ::cuda::std::initializer_list<_Tp> __ilist, const _Env& __env = {})
{
  return buffer<_Tp, _FirstProperty, _RestProperties...>{
    __stream, ::cuda::std::forward<_Resource>(__mr), __ilist, __env};
}

_CCCL_TEMPLATE(class _Tp, class _Resource, class _Env = ::cuda::std::execution::env<>)
_CCCL_REQUIRES(::cuda::mr::synchronous_resource<::cuda::std::decay_t<_Resource>>
                 _CCCL_AND ::cuda::mr::__has_default_queries<::cuda::std::decay_t<_Resource>>)
auto make_buffer(
  stream_ref __stream, _Resource&& __mr, ::cuda::std::initializer_list<_Tp> __ilist, const _Env& __env = {})
{
  using __buffer_type = __buffer_type_for_props<_Tp, typename ::cuda::std::decay_t<_Resource>::default_queries>;
  return __buffer_type{__stream, ::cuda::std::forward<_Resource>(__mr), __ilist, __env};
}

// Range make function for ranges
_CCCL_TEMPLATE(class _Tp,
               class _FirstProperty,
               class... _RestProperties,
               class _Resource,
               class _Range,
               class _Env = ::cuda::std::execution::env<>)
_CCCL_REQUIRES(
  ::cuda::mr::synchronous_resource_with<::cuda::std::decay_t<_Resource>, _FirstProperty, _RestProperties...> _CCCL_AND
    __buffer_compatible_env<_Env> _CCCL_AND ::cuda::std::ranges::forward_range<_Range>)
buffer<_Tp, _FirstProperty, _RestProperties...>
make_buffer(stream_ref __stream, _Resource&& __mr, _Range&& __range, const _Env& __env = {})
{
  return buffer<_Tp, _FirstProperty, _RestProperties...>{
    __stream, ::cuda::std::forward<_Resource>(__mr), ::cuda::std::forward<_Range>(__range), __env};
}

_CCCL_TEMPLATE(class _Tp, class _Resource, class _Range, class _Env = ::cuda::std::execution::env<>)
_CCCL_REQUIRES(
  ::cuda::mr::synchronous_resource<::cuda::std::decay_t<_Resource>> _CCCL_AND ::cuda::mr::__has_default_queries<
    ::cuda::std::decay_t<_Resource>> _CCCL_AND ::cuda::std::ranges::forward_range<_Range>)
auto make_buffer(stream_ref __stream, _Resource&& __mr, _Range&& __range, const _Env& __env = {})
{
  using __buffer_type = __buffer_type_for_props<_Tp, typename ::cuda::std::decay_t<_Resource>::default_queries>;
  return __buffer_type{__stream, ::cuda::std::forward<_Resource>(__mr), ::cuda::std::forward<_Range>(__range), __env};
}
_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif //_CUDA___CONTAINER_BUFFER_H
