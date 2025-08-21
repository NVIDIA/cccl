//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINER_ASYNC_BUFFER__
#define __CUDAX__CONTAINER_ASYNC_BUFFER__

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/copy.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>

#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__memory_resource/properties.h>
#include <cuda/__memory_resource/resource_ref.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__algorithm/equal.h>
#include <cuda/std/__algorithm/fill.h>
#include <cuda/std/__algorithm/lexicographical_compare.h>
#include <cuda/std/__algorithm/move_backward.h>
#include <cuda/std/__algorithm/rotate.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/reverse_iterator.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__memory/temporary_buffer.h>
#include <cuda/std/__memory/uninitialized_algorithms.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/unwrap_end.h>
#include <cuda/std/__type_traits/is_nothrow_move_assignable.h>
#include <cuda/std/__type_traits/is_swappable.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>
#include <cuda/std/detail/libcxx/include/stdexcept>
#include <cuda/std/initializer_list>
#include <cuda/std/limits>

#include <cuda/experimental/__container/heterogeneous_iterator.cuh>
#include <cuda/experimental/__container/uninitialized_async_buffer.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/policy.cuh>
#include <cuda/experimental/__launch/host_launch.cuh>
#include <cuda/experimental/__memory_resource/any_resource.cuh>
#include <cuda/experimental/__memory_resource/properties.cuh>
#include <cuda/experimental/__utility/ensure_current_device.cuh>
#include <cuda/experimental/__utility/select_execution_space.cuh>

#include <cuda/std/__cccl/prologue.h>

//! @file The \c async_buffer class provides a container of contiguous memory
namespace cuda::experimental
{

//! @rst
//! .. _cudax-containers-async-vector:
//!
//! async_buffer
//! -------------
//!
//! ``async_buffer`` is a container that provides resizable typed storage allocated from a given :ref:`memory resource
//! <libcudacxx-extended-api-memory-resources-resource>`. It handles alignment, release and growth of the allocation.
//! The elements are initialized during construction, which may require a kernel launch.
//!
//! In addition to being type-safe, ``async_buffer`` also takes a set of :ref:`properties
//! <libcudacxx-extended-api-memory-resources-properties>` to ensure that e.g. execution space constraints are checked
//! at compile time. However, only stateless properties can be forwarded. To use a stateful property,
//! implement :ref:`get_property(const async_buffer&, Property) <libcudacxx-extended-api-memory-resources-properties>`.
//!
//! @endrst
//! @tparam _Tp the type to be stored in the buffer
//! @tparam _Properties... The properties the allocated memory satisfies
template <class _Tp, class... _Properties>
class async_buffer
{
public:
  using value_type             = _Tp;
  using reference              = _Tp&;
  using const_reference        = const _Tp&;
  using pointer                = _Tp*;
  using const_pointer          = const _Tp*;
  using iterator               = heterogeneous_iterator<_Tp, _Properties...>;
  using const_iterator         = heterogeneous_iterator<const _Tp, _Properties...>;
  using reverse_iterator       = ::cuda::std::reverse_iterator<iterator>;
  using const_reverse_iterator = ::cuda::std::reverse_iterator<const_iterator>;
  using size_type              = ::cuda::std::size_t;
  using difference_type        = ::cuda::std::ptrdiff_t;

  using __env_t          = ::cuda::experimental::env_t<_Properties...>;
  using __policy_t       = ::cuda::experimental::execution::any_execution_policy;
  using __buffer_t       = ::cuda::experimental::uninitialized_async_buffer<_Tp, _Properties...>;
  using __resource_t     = ::cuda::experimental::any_resource<_Properties...>;
  using __resource_ref_t = ::cuda::mr::resource_ref<_Properties...>;

  template <class, class...>
  friend class async_buffer;

  // For now we require trivially copyable type to simplify the implementation
  static_assert(::cuda::std::is_trivially_copyable_v<_Tp>,
                "cuda::experimental::async_buffer requires T to be trivially copyable.");

  // At least one of the properties must signal an execution space
  static_assert(::cuda::mr::__contains_execution_space_property<_Properties...>,
                "The properties of cuda::experimental::async_buffer must contain at least one execution space "
                "property!");

  //! @brief Convenience shortcut to detect the execution space of the async_buffer
  static constexpr bool __is_host_only = __select_execution_space<_Properties...> == _ExecutionSpace::__host;

private:
  __buffer_t __buf_;

  //! @brief Helper to check container is compatible with this async_buffer
  template <class _Range>
  static constexpr bool __compatible_range = (::cuda::std::ranges::__container_compatible_range<_Range, _Tp>);

  //! @brief Helper to check whether a different async_buffer still satisfies all properties of this one
  template <class... _OtherProperties>
  static constexpr bool __properties_match =
    ::cuda::std::__type_set_contains_v<::cuda::std::__make_type_set<_OtherProperties...>, _Properties...>;

  //! @brief Helper to determine what cudaMemcpyKind we need to copy data from another async_buffer with different
  //!        properties. Needed for compilers that have issues handling packs in constraints
  template <class... _OtherProperties>
  static constexpr cudaMemcpyKind __transfer_kind =
    __select_execution_space<_OtherProperties...> == _ExecutionSpace::__host
      ? (__is_host_only ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice)
      : (__is_host_only ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);

  //! @brief Helper to return an resource_ref to the currently used resource. Used to grow the async_buffer
  __resource_ref_t __borrow_resource() const noexcept
  {
    return const_cast<__resource_t&>(__buf_.memory_resource());
  }

  //! @brief Copies \p __count elements from `[__first, __last)` to \p __dest, where \p __first and \p __dest reside in
  //! the different memory spaces
  //! @param __first Pointer to the start of the input segment.
  //! @param __last Pointer to the end of the input segment.
  //! @param __dest Pointer to the start of the output segment.
  //! @param __count The number of elements to be copied.
  //! @note This function is inherently asynchronous. We need to ensure that the memory pointed to by \p __first and
  //! \p __last lives long enough
  template <class _Iter>
  _CCCL_HIDE_FROM_ABI void __copy_cross(_Iter __first, [[maybe_unused]] _Iter __last, pointer __dest, size_type __count)
  {
    if (__count == 0)
    {
      return;
    }

    static_assert(::cuda::std::contiguous_iterator<_Iter>, "Non contiguous iterators are not supported");
    // TODO use batched memcpy for non-contiguous iterators, it allows to specify stream ordered access
    ::cuda::__driver::__memcpyAsync(
      __dest, ::cuda::std::to_address(__first), sizeof(_Tp) * __count, __buf_.stream().get());
  }

  //! @brief Value-initializes elements in the range `[__first, __first + __count)`.
  //! @param __first Pointer to the first element to be initialized.
  //! @param __count The number of elements to be initialized.
  _CCCL_HIDE_FROM_ABI void __value_initialize_n(pointer __first, size_type __count)
  {
    if (__count == 0)
    {
      return;
    }

    if constexpr (__is_host_only)
    {
      ::cuda::experimental::host_launch(
        __buf_.stream(), ::cuda::std::uninitialized_value_construct_n<pointer, size_type>, __first, __count);
    }
    else
    {
      ::cuda::experimental::__ensure_current_device __guard(__buf_.stream());
      thrust::fill_n(thrust::cuda::par_nosync.on(__buf_.stream().get()), __first, __count, _Tp());
    }
  }

  //! @brief Copy-constructs elements in the range `[__first, __first + __count)`.
  //! @param __first Pointer to the first element to be initialized.
  //! @param __count The number of elements to be initialized.
  _CCCL_HIDE_FROM_ABI void __fill_n(pointer __first, size_type __count, const _Tp& __value)
  {
    if (__count == 0)
    {
      return;
    }

    if constexpr (__is_host_only)
    {
      ::cuda::experimental::host_launch(
        __buf_.stream(), ::cuda::std::uninitialized_fill_n<pointer, size_type, _Tp>, __first, __count, __value);
    }
    else
    {
      ::cuda::experimental::__ensure_current_device __guard(__buf_.stream());
      thrust::fill_n(thrust::cuda::par_nosync.on(__buf_.stream().get()), __first, __count, __value);
    }
  }

public:
  //! @addtogroup construction
  //! @{

  //! @brief Copy-constructs from a async_buffer
  //! @param __other The other async_buffer.
  _CCCL_HIDE_FROM_ABI async_buffer(const async_buffer& __other)
      : __buf_(__other.memory_resource(), __other.stream(), __other.size())
  {
    this->__copy_cross<const_pointer>(
      __other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin(), __other.size());
  }

  //! @brief Move-constructs from a async_buffer
  //! @param __other The other async_buffer. After move construction, the other buffer can only be assigned to or
  //! destroyed.
  _CCCL_HIDE_FROM_ABI async_buffer(async_buffer&& __other) noexcept
      : __buf_(::cuda::std::move(__other.__buf_))
  {}

  //! @brief Copy-constructs from a async_buffer with matching properties
  //! @param __other The other async_buffer.
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HIDE_FROM_ABI explicit async_buffer(const async_buffer<_Tp, _OtherProperties...>& __other)
      : __buf_(__other.memory_resource(), __other.stream(), __other.size())
  {
    this->__copy_cross<const_pointer>(
      __other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin(), __other.size());
  }

  //! @brief Move-constructs from a async_buffer with matching properties
  //! @param __other The other async_buffer. After move construction, the other buffer can only be assigned to or
  //! destroyed.
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HIDE_FROM_ABI explicit async_buffer(async_buffer<_Tp, _OtherProperties...>&& __other) noexcept
      : __buf_(::cuda::std::move(__other.__buf_))
  {}

  //! @brief Constructs an empty async_buffer using an environment
  //! @param __env The environment providing the needed information
  //! @note No memory is allocated.
  _CCCL_HIDE_FROM_ABI async_buffer(const __env_t& __env)
      : async_buffer(__env, 0, ::cuda::experimental::no_init)
  {}

  //! @brief Constructs a async_buffer of size \p __size using a memory resource and value-initializes \p __size
  //! elements
  //! @param __env The environment used to query the memory resource.
  //! @param __size The size of the async_buffer. Defaults to zero
  //! @note If `__size == 0` then no memory is allocated.
  _CCCL_HIDE_FROM_ABI explicit async_buffer(const __env_t& __env, const size_type __size)
      : async_buffer(__env, __size, ::cuda::experimental::no_init)
  {
    this->__value_initialize_n(__unwrapped_begin(), __size);
  }

  //! @brief Constructs a async_buffer of size \p __size using a memory resource and copy-constructs \p __size elements
  //! from \p __value
  //! @param __env The environment used to query the memory resource.
  //! @param __size The size of the async_buffer.
  //! @param __value The value all elements are copied from.
  //! @note If `__size == 0` then no memory is allocated.
  _CCCL_HIDE_FROM_ABI explicit async_buffer(const __env_t& __env, const size_type __size, const _Tp& __value)
      : async_buffer(__env, __size, ::cuda::experimental::no_init)
  {
    this->__fill_n(__unwrapped_begin(), __size, __value);
  }

  //! @brief Constructs a async_buffer of size \p __size using a memory and leaves all elements uninitialized
  //! @param __env The environment used to query the memory resource.
  //! @param __size The size of the async_buffer.
  //! @warning This constructor does *NOT* initialize any elements. It is the user's responsibility to ensure that the
  //! elements within `[vec.begin(), vec.end())` are properly initialized, e.g with `cuda::std::uninitialized_copy`.
  //! At the destruction of the \c async_buffer all elements in the range `[vec.begin(), vec.end())` will be destroyed.
  _CCCL_HIDE_FROM_ABI explicit async_buffer(
    const __env_t& __env, const size_type __size, ::cuda::experimental::no_init_t)
      : __buf_(::cuda::mr::get_memory_resource(__env), ::cuda::get_stream(__env), __size)
  {}

  //! @brief Constructs a async_buffer using a memory resource and copy-constructs all elements from the forward range
  //! ``[__first, __last)``
  //! @param __env The environment used to query the memory resource.
  //! @param __first The start of the input sequence.
  //! @param __last The end of the input sequence.
  //! @note If `__first == __last` then no memory is allocated
  _CCCL_TEMPLATE(class _Iter)
  _CCCL_REQUIRES(::cuda::std::__is_cpp17_forward_iterator<_Iter>::value)
  _CCCL_HIDE_FROM_ABI async_buffer(const __env_t& __env, _Iter __first, _Iter __last)
      : async_buffer(
          __env, static_cast<size_type>(::cuda::std::distance(__first, __last)), ::cuda::experimental::no_init)
  {
    this->__copy_cross<_Iter>(__first, __last, __unwrapped_begin(), __buf_.size());
  }

  //! @brief Constructs a async_buffer using a memory resource and copy-constructs all elements from \p __ilist
  //! @param __env The environment used to query the memory resource.
  //! @param __ilist The initializer_list being copied into the async_buffer.
  //! @note If `__ilist.size() == 0` then no memory is allocated
  _CCCL_HIDE_FROM_ABI async_buffer(const __env_t& __env, ::cuda::std::initializer_list<_Tp> __ilist)
      : async_buffer(__env, __ilist.size(), ::cuda::experimental::no_init)
  {
    this->__copy_cross(__ilist.begin(), __ilist.end(), __unwrapped_begin(), __buf_.size());
  }

  //! @brief Constructs a async_buffer using a memory resource and an input range
  //! @param __env The environment used to query the memory resource.
  //! @param __range The input range to be moved into the async_buffer.
  //! @note If `__range.size() == 0` then no memory is allocated.
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND ::cuda::std::ranges::forward_range<_Range>
                   _CCCL_AND ::cuda::std::ranges::sized_range<_Range>)
  _CCCL_HIDE_FROM_ABI async_buffer(const __env_t& __env, _Range&& __range)
      : async_buffer(__env, static_cast<size_type>(::cuda::std::ranges::size(__range)), ::cuda::experimental::no_init)
  {
    using _Iter = ::cuda::std::ranges::iterator_t<_Range>;
    this->__copy_cross<_Iter>(
      ::cuda::std::ranges::begin(__range),
      ::cuda::std::ranges::__unwrap_end(__range),
      __unwrapped_begin(),
      __buf_.size());
  }

#ifndef _CCCL_DOXYGEN_INVOKED // doxygen conflates the overloads
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND ::cuda::std::ranges::forward_range<_Range> _CCCL_AND(
    !::cuda::std::ranges::sized_range<_Range>))
  _CCCL_HIDE_FROM_ABI async_buffer(const __env_t& __env, _Range&& __range)
      : async_buffer(__env,
                     static_cast<size_type>(::cuda::std::ranges::distance(
                       ::cuda::std::ranges::begin(__range), ::cuda::std::ranges::end(__range))),
                     ::cuda::experimental::no_init)
  {
    using _Iter = ::cuda::std::ranges::iterator_t<_Range>;
    this->__copy_cross<_Iter>(
      ::cuda::std::ranges::begin(__range),
      ::cuda::std::ranges::__unwrap_end(__range),
      __unwrapped_begin(),
      __buf_.size());
  }
#endif // _CCCL_DOXYGEN_INVOKED
  //! @}

  //! @addtogroup iterators
  //! @{
  //! @brief Returns an iterator to the first element of the async_buffer. If the async_buffer is empty, the returned
  //! iterator will be equal to end().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI iterator begin() noexcept
  {
    return iterator{__buf_.data()};
  }

  //! @brief Returns an immutable iterator to the first element of the async_buffer. If the async_buffer is empty, the
  //! returned iterator will be equal to end().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_iterator begin() const noexcept
  {
    return const_iterator{__buf_.data()};
  }

  //! @brief Returns an immutable iterator to the first element of the async_buffer. If the async_buffer is empty, the
  //! returned iterator will be equal to end().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_iterator cbegin() const noexcept
  {
    return const_iterator{__buf_.data()};
  }

  //! @brief Returns an iterator to the element following the last element of the async_buffer. This element acts as a
  //! placeholder; attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI iterator end() noexcept
  {
    return iterator{__buf_.data() + __buf_.size()};
  }

  //! @brief Returns an immutable iterator to the element following the last element of the async_buffer. This element
  //! acts as a placeholder; attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_iterator end() const noexcept
  {
    return const_iterator{__buf_.data() + __buf_.size()};
  }

  //! @brief Returns an immutable iterator to the element following the last element of the async_buffer. This element
  //! acts as a placeholder; attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_iterator cend() const noexcept
  {
    return const_iterator{__buf_.data() + __buf_.size()};
  }

  //! @brief Returns a reverse iterator to the first element of the reversed async_buffer. It corresponds to the last
  //! element of the non-reversed async_buffer. If the async_buffer is empty, the returned iterator is equal to rend().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI reverse_iterator rbegin() noexcept
  {
    return reverse_iterator{end()};
  }

  //! @brief Returns an immutable reverse iterator to the first element of the reversed async_buffer. It corresponds to
  //! the last element of the non-reversed async_buffer. If the async_buffer is empty, the returned iterator is equal to
  //! rend().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reverse_iterator rbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns an immutable reverse iterator to the first element of the reversed async_buffer. It corresponds to
  //! the last element of the non-reversed async_buffer. If the async_buffer is empty, the returned iterator is equal to
  //! rend().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reverse_iterator crbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns a reverse iterator to the element following the last element of the reversed async_buffer. It
  //! corresponds to the element preceding the first element of the non-reversed async_buffer. This element acts as a
  //! placeholder, attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI reverse_iterator rend() noexcept
  {
    return reverse_iterator{begin()};
  }

  //! @brief Returns an immutable reverse iterator to the element following the last element of the reversed
  //! async_buffer. It corresponds to the element preceding the first element of the non-reversed async_buffer. This
  //! element acts as a placeholder, attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reverse_iterator rend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns an immutable reverse iterator to the element following the last element of the reversed
  //! async_buffer. It corresponds to the element preceding the first element of the non-reversed async_buffer. This
  //! element acts as a placeholder, attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reverse_iterator crend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns a pointer to the first element of the async_buffer. If the async_buffer has not allocated memory
  //! the pointer will be null.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI pointer data() noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a pointer to the first element of the async_buffer. If the async_buffer has not allocated memory
  //! the pointer will be null.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_pointer data() const noexcept
  {
    return __buf_.data();
  }

#ifndef _CCCL_DOXYGEN_INVOKED
  //! @brief Returns a pointer to the first element of the async_buffer. If the async_buffer is empty, the returned
  //! pointer will be null.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI pointer __unwrapped_begin() noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a const pointer to the first element of the async_buffer. If the async_buffer is empty, the
  //! returned pointer will be null.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_pointer __unwrapped_begin() const noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a pointer to the element following the last element of the async_buffer. This element acts as a
  //! placeholder; attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI pointer __unwrapped_end() noexcept
  {
    return __buf_.data() + __buf_.size();
  }

  //! @brief Returns a const pointer to the element following the last element of the async_buffer. This element acts as
  //! a placeholder; attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_pointer __unwrapped_end() const noexcept
  {
    return __buf_.data() + __buf_.size();
  }
#endif // _CCCL_DOXYGEN_INVOKED

  //! @}

  //! @addtogroup access
  //! @brief Returns a reference to the \p __n 'th element of the async_vector
  //! @param __n The index of the element we want to access
  //! @note Does not synchronize with the stored stream
  [[nodiscard]] _CCCL_HIDE_FROM_ABI reference get_unsynchronized(const size_type __n) noexcept
  {
    _CCCL_ASSERT(__n < __buf_.size(), "cuda::experimental::async_buffer::get_unsynchronized out of range!");
    return begin()[__n];
  }

  //! @brief Returns a reference to the \p __n 'th element of the async_vector
  //! @param __n The index of the element we want to access
  //! @note Does not synchronize with the stored stream
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reference get_unsynchronized(const size_type __n) const noexcept
  {
    _CCCL_ASSERT(__n < __buf_.size(), "cuda::experimental::async_buffer::get_unsynchronized out of range!");
    return begin()[__n];
  }

  //! @}

  //! @addtogroup size
  //! @{
  //! @brief Returns the current number of elements stored in the async_buffer.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI size_type size() const noexcept
  {
    return __buf_.size();
  }

  //! @brief Returns true if the async_buffer is empty.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI bool empty() const noexcept
  {
    return __buf_.size() == 0;
  }
  //! @}

  //! @rst
  //! Returns a \c const reference to the :ref:`any_resource <cudax-memory-resource-any-resource>`
  //! that holds the memory resource used to allocate the async_buffer
  //! @endrst
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const __resource_t& memory_resource() const noexcept
  {
    return __buf_.memory_resource();
  }

  //! @brief Returns the stored stream
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr stream_ref stream() const noexcept
  {
    return __buf_.stream();
  }

  //! @brief Replaces the stored stream
  //! @param __new_stream the new stream
  //! @note Always synchronizes with the old stream
  _CCCL_HIDE_FROM_ABI constexpr void set_stream(stream_ref __new_stream)
  {
    __buf_.set_stream(__new_stream);
  }

  //! @brief Replaces the stored stream
  //! @param __new_stream the new stream
  //! @warning This does not synchronize between \p __new_stream and the current stream. It is the user's responsibility
  //! to ensure proper stream order going forward
  _CCCL_HIDE_FROM_ABI constexpr void set_stream_unsynchronized(stream_ref __new_stream) noexcept
  {
    __buf_.set_stream_unsynchronized(__new_stream);
  }

  //! @brief Move assignment operator
  //! @param __other The other async_buffer. After move assignment, the other buffer can only be assigned to or
  //! destroyed.
  _CCCL_HIDE_FROM_ABI void operator=(async_buffer&& __other)
  {
    __buf_ = ::cuda::std::move(__other.__buf_);
  }

  //! @brief Swaps the contents of a async_buffer with those of \p __other
  //! @param __other The other async_buffer.
  _CCCL_HIDE_FROM_ABI void swap(async_buffer& __other) noexcept
  {
    ::cuda::std::swap(__buf_, __other.__buf_);
  }

  //! @brief Swaps the contents of two async_buffers
  //! @param __lhs One async_buffer.
  //! @param __rhs The other async_buffer.
  _CCCL_HIDE_FROM_ABI friend void swap(async_buffer& __lhs, async_buffer& __rhs) noexcept
  {
    __lhs.swap(__rhs);
  }

  //! @brief Destroys the async_buffer, deallocates the buffer and destroys the memory resource
  //! @warning After this explicit destroy call, the buffer can only be assigned to or destroyed.
  _CCCL_HIDE_FROM_ABI void destroy()
  {
    __buf_.destroy();
  }

  //! @brief Causes the buffer to be treated as a span when passed to cudax::launch.
  //! @pre The buffer must have the cuda::mr::device_accessible property.
  template <class _DeviceAccessible = device_accessible>
  [[nodiscard]] _CCCL_HIDE_FROM_ABI friend auto
  transform_device_argument(::cuda::stream_ref, async_buffer& __self) noexcept
    _CCCL_TRAILING_REQUIRES(::cuda::std::span<_Tp>)(::cuda::std::__is_included_in_v<_DeviceAccessible, _Properties...>)
  {
    // TODO add auto synchronization
    return {__self.__unwrapped_begin(), __self.size()};
  }

  //! @brief Causes the buffer to be treated as a span when passed to cudax::launch
  //! @pre The buffer must have the cuda::mr::device_accessible property.
  template <class _DeviceAccessible = device_accessible>
  [[nodiscard]] _CCCL_HIDE_FROM_ABI friend auto
  transform_device_argument(::cuda::stream_ref, const async_buffer& __self) noexcept _CCCL_TRAILING_REQUIRES(
    ::cuda::std::span<const _Tp>)(::cuda::std::__is_included_in_v<_DeviceAccessible, _Properties...>)
  {
    // TODO add auto synchronization
    return {__self.__unwrapped_begin(), __self.size()};
  }

  //! @brief Forwards the passed properties
  _CCCL_TEMPLATE(class _Property)
  _CCCL_REQUIRES((!property_with_value<_Property>) _CCCL_AND ::cuda::std::__is_included_in_v<_Property, _Properties...>)
  _CCCL_HIDE_FROM_ABI friend void get_property(const async_buffer&, _Property) noexcept {}
};

template <class _Tp>
using async_device_buffer = async_buffer<_Tp, ::cuda::mr::device_accessible>;

template <class _Tp>
using async_host_buffer = async_buffer<_Tp, ::cuda::mr::host_accessible>;

template <class _Tp, class _PropsList>
using __buffer_type_for_props =
  typename ::cuda::std::remove_reference_t<_PropsList>::template rebind<async_buffer, _Tp>;

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

template <class _Tp, class... _TargetProperties, class... _SourceProperties>
async_buffer<_Tp, _TargetProperties...> make_async_buffer(
  stream_ref __stream, any_resource<_TargetProperties...> __mr, const async_buffer<_Tp, _SourceProperties...>& __source)
{
  env_t<_TargetProperties...> __env{__mr, __stream};
  async_buffer<_Tp, _TargetProperties...> __res{__env, __source.size(), no_init};

  __copy_cross_buffers(__stream, __res, __source);

  return __res;
}

_CCCL_TEMPLATE(class _Tp, class _Resource, class... _SourceProperties)
_CCCL_REQUIRES(::cuda::mr::resource<_Resource> _CCCL_AND __has_default_queries<_Resource>)
auto make_async_buffer(stream_ref __stream, _Resource&& __mr, const async_buffer<_Tp, _SourceProperties...>& __source)
{
  using __buffer_type = __buffer_type_for_props<_Tp, typename ::cuda::std::decay_t<_Resource>::default_queries>;
  typename __buffer_type::__env_t __env{__mr, __stream};
  auto __res = __buffer_type{__env, __source.size(), uninit};

  __copy_cross_buffers(__stream, __res, __source);

  return __res;
}

// Empty buffer make function
template <class _Tp, class... _Properties>
async_buffer<_Tp, _Properties...> make_async_buffer(stream_ref __stream, any_resource<_Properties...> __mr)
{
  env_t<_Properties...> __env{__mr, __stream};
  return async_buffer<_Tp, _Properties...>{__env};
}

_CCCL_TEMPLATE(class _Tp, class _Resource)
_CCCL_REQUIRES(::cuda::mr::resource<_Resource> _CCCL_AND __has_default_queries<_Resource>)
auto make_async_buffer(stream_ref __stream, _Resource&& __mr)
{
  using __buffer_type = __buffer_type_for_props<_Tp, typename ::cuda::std::decay_t<_Resource>::default_queries>;
  typename __buffer_type::__env_t __env{__mr, __stream};
  return __buffer_type{__env};
}

// Size-only make function
template <class _Tp, class... _Properties>
async_buffer<_Tp, _Properties...>
make_async_buffer(stream_ref __stream, any_resource<_Properties...> __mr, size_t __size)
{
  env_t<_Properties...> __env{__mr, __stream};
  return async_buffer<_Tp, _Properties...>{__env, __size};
}

_CCCL_TEMPLATE(class _Tp, class _Resource)
_CCCL_REQUIRES(::cuda::mr::resource<_Resource> _CCCL_AND __has_default_queries<_Resource>)
auto make_async_buffer(stream_ref __stream, _Resource&& __mr, size_t __size)
{
  using __buffer_type = __buffer_type_for_props<_Tp, typename ::cuda::std::decay_t<_Resource>::default_queries>;
  typename __buffer_type::__env_t __env{__mr, __stream};
  return __buffer_type{__env, __size};
}

// Size and value make function
template <class _Tp, class... _Properties>
async_buffer<_Tp, _Properties...>
make_async_buffer(stream_ref __stream, any_resource<_Properties...> __mr, size_t __size, const _Tp& __value)
{
  env_t<_Properties...> __env{__mr, __stream};
  return async_buffer<_Tp, _Properties...>{__env, __size, __value};
}

_CCCL_TEMPLATE(class _Tp, class _Resource)
_CCCL_REQUIRES(::cuda::mr::resource<_Resource> _CCCL_AND __has_default_queries<_Resource>)
auto make_async_buffer(stream_ref __stream, _Resource&& __mr, size_t __size, const _Tp& __value)
{
  using __buffer_type = __buffer_type_for_props<_Tp, typename ::cuda::std::decay_t<_Resource>::default_queries>;
  typename __buffer_type::__env_t __env{__mr, __stream};
  return __buffer_type{__env, __size, __value};
}

// Size with no initialization make function
template <class _Tp, class... _Properties>
async_buffer<_Tp, _Properties...> make_async_buffer(
  stream_ref __stream, any_resource<_Properties...> __mr, size_t __size, ::cuda::experimental::no_init_t)
{
  env_t<_Properties...> __env{__mr, __stream};
  return async_buffer<_Tp, _Properties...>{__env, __size, ::cuda::experimental::no_init};
}

_CCCL_TEMPLATE(class _Tp, class _Resource)
_CCCL_REQUIRES(::cuda::mr::resource<_Resource> _CCCL_AND __has_default_queries<_Resource>)
auto make_async_buffer(stream_ref __stream, _Resource&& __mr, size_t __size, ::cuda::experimental::no_init_t)
{
  using __buffer_type = __buffer_type_for_props<_Tp, typename ::cuda::std::decay_t<_Resource>::default_queries>;
  typename __buffer_type::__env_t __env{__mr, __stream};
  return __buffer_type{__env, __size, ::cuda::experimental::no_init};
}

// Iterator range make function
_CCCL_TEMPLATE(class _Tp, class... _Properties, class _Iter)
_CCCL_REQUIRES(::cuda::std::__is_cpp17_forward_iterator<_Iter>::value)
async_buffer<_Tp, _Properties...>
make_async_buffer(stream_ref __stream, any_resource<_Properties...> __mr, _Iter __first, _Iter __last)
{
  env_t<_Properties...> __env{__mr, __stream};
  return async_buffer<_Tp, _Properties...>{__env, __first, __last};
}

_CCCL_TEMPLATE(class _Tp, class _Resource, class _Iter)
_CCCL_REQUIRES(::cuda::mr::resource<_Resource> _CCCL_AND
                 __has_default_queries<_Resource> _CCCL_AND ::cuda::std::__is_cpp17_forward_iterator<_Iter>::value)
auto make_async_buffer(stream_ref __stream, _Resource&& __mr, _Iter __first, _Iter __last)
{
  using __buffer_type = __buffer_type_for_props<_Tp, typename ::cuda::std::decay_t<_Resource>::default_queries>;
  typename __buffer_type::__env_t __env{__mr, __stream};
  return __buffer_type{__env, __first, __last};
}

// Initializer list make function
template <class _Tp, class... _Properties>
async_buffer<_Tp, _Properties...>
make_async_buffer(stream_ref __stream, any_resource<_Properties...> __mr, ::cuda::std::initializer_list<_Tp> __ilist)
{
  env_t<_Properties...> __env{__mr, __stream};
  return async_buffer<_Tp, _Properties...>{__env, __ilist};
}

_CCCL_TEMPLATE(class _Tp, class _Resource)
_CCCL_REQUIRES(::cuda::mr::resource<_Resource> _CCCL_AND __has_default_queries<_Resource>)
auto make_async_buffer(stream_ref __stream, _Resource&& __mr, ::cuda::std::initializer_list<_Tp> __ilist)
{
  using __buffer_type = __buffer_type_for_props<_Tp, typename ::cuda::std::decay_t<_Resource>::default_queries>;
  typename __buffer_type::__env_t __env{__mr, __stream};
  return __buffer_type{__env, __ilist};
}

// Range make function for ranges
_CCCL_TEMPLATE(class _Tp, class... _Properties, class _Range)
_CCCL_REQUIRES(::cuda::std::ranges::forward_range<_Range>)
async_buffer<_Tp, _Properties...>
make_async_buffer(stream_ref __stream, any_resource<_Properties...> __mr, _Range&& __range)
{
  env_t<_Properties...> __env{__mr, __stream};
  return async_buffer<_Tp, _Properties...>{__env, ::cuda::std::forward<_Range>(__range)};
}

_CCCL_TEMPLATE(class _Tp, class _Resource, class _Range)
_CCCL_REQUIRES(::cuda::mr::resource<_Resource> _CCCL_AND
                 __has_default_queries<_Resource> _CCCL_AND ::cuda::std::ranges::forward_range<_Range>)
auto make_async_buffer(stream_ref __stream, _Resource&& __mr, _Range&& __range)
{
  using __buffer_type = __buffer_type_for_props<_Tp, typename ::cuda::std::decay_t<_Resource>::default_queries>;
  typename __buffer_type::__env_t __env{__mr, __stream};
  return __buffer_type{__env, ::cuda::std::forward<_Range>(__range)};
}

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //__CUDAX__CONTAINER_ASYNC_BUFFER__
