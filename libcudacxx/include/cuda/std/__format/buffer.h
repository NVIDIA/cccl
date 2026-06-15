//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_BUFFER_H
#define _CUDA_STD___FORMAT_BUFFER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__algorithm/copy_n.h>
#include <cuda/std/__algorithm/fill_n.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/transform.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__format/concepts.h>
#include <cuda/std/__fwd/format.h>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__iterator/back_insert_iterator.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/wrap_iter.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/allocate_at_least.h>
#include <cuda/std/__memory/allocator.h>
#include <cuda/std/__memory/destruct_n.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/exception_guard.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// A helper to limit the total size of code units written.
class __fmt_max_output_size
{
  size_t __max_size_;
  // The code units that would have been written if there was no limit.
  // format_to_n returns this value.
  size_t __code_units_written_{0};

public:
  _CCCL_API constexpr explicit __fmt_max_output_size(size_t __max_size)
      : __max_size_{__max_size}
  {}

  // This function adjusts the size of a (bulk) write operations. It ensures the
  // number of code units written by a __fmt_output_buffer never exceeds
  // __max_size_ code units.
  [[nodiscard]] _CCCL_API constexpr size_t __write_request(size_t __code_units)
  {
    const auto __result =
      (__code_units_written_ < __max_size_) ? ::cuda::std::min(__code_units, __max_size_ - __code_units_written_) : 0;
    __code_units_written_ += __code_units;
    return __result;
  }

  [[nodiscard]] _CCCL_API constexpr size_t __code_units_written() const noexcept
  {
    return __code_units_written_;
  }
};

/// A "buffer" that handles writing to the proper iterator.
///
/// This helper is used together with the @ref __back_insert_iterator to offer
/// type-erasure for the formatting functions. This reduces the number to
/// template instantiations.
///
/// The design is the following:
/// - There is an external object that connects the buffer to the output.
/// - This buffer object:
///   - inherits publicly from this class.
///   - has a static or dynamic buffer.
///   - has a static member function to make space in its buffer write
///     operations. This can be done by increasing the size of the internal
///     buffer or by writing the contents of the buffer to the output iterator.
///
///     This member function is a constructor argument, so its name is not
///     fixed. The code uses the name __prepare_write.
/// - The number of output code units can be limited by a __fmt_max_output_size
///   object. This is used in format_to_n This object:
///   - Contains the maximum number of code units to be written.
///   - Contains the number of code units that are requested to be written.
///     This number is returned to the user of format_to_n.
///   - The write functions call the object's __request_write member function.
///     This function:
///     - Updates the number of code units that are requested to be written.
///     - Returns the number of code units that can be written without
///       exceeding the maximum number of code units to be written.
///
/// Documentation for the buffer usage members:
/// - __ptr_
///   The start of the buffer.
/// - __capacity_
///   The number of code units that can be written. This means
///   [__ptr_, __ptr_ + __capacity_) is a valid range to write to.
/// - __size_
///   The number of code units written in the buffer. The next code unit will
///   be written at __ptr_ + __size_. This __size_ may NOT contain the total
///   number of code units written by the __fmt_output_buffer. Whether or not it
///   does depends on the sub-class used. Typically the total number of code
///   units written is not interesting. It is interesting for format_to_n which
///   has its own way to track this number.
///
/// Documentation for the modifying buffer operations:
/// The subclasses have a function with the following signature:
///
///   static void __prepare_write(
///     __fmt_output_buffer<_CharT>& __buffer, size_t __code_units);
///
/// This function is called when a write function writes more code units than
/// the buffer's available space. When an __max_output_size object is provided
/// the number of code units is the number of code units returned from
/// __max_output_size::__request_write function.
///
/// - The __buffer contains *this. Since the class containing this function
///   inherits from __fmt_output_buffer it's safe to cast it to the subclass being
///   used.
/// - The __code_units is the number of code units the caller will write + 1.
///   - This value does not take the available space of the buffer into account.
///   - The push_back function is more efficient when writing before resizing,
///     this means the buffer should always have room for one code unit. Hence
///     the + 1 is the size.
/// - When the function returns there is room for at least one additional code
///   unit. There is no requirement there is room for __code_units code units:
///   - The class has some "bulk" operations. For example, __copy which copies
///     the contents of a basic_string_view to the output. If the sub-class has
///     a fixed size buffer the size of the basic_string_view may be larger
///     than the buffer. In that case it's impossible to honor the requested
///     size.
///   - When the buffer has room for at least one code unit the function may be
///     a no-op.
/// - When the function makes space for more code units it uses one for these
///   functions to signal the change:
///   - __buffer_flushed()
///     - This function is typically used for a fixed sized buffer.
///     - The current contents of [__ptr_, __ptr_ + __size_) have been
///       processed.
///     - __ptr_ remains unchanged.
///     - __capacity_ remains unchanged.
///     - __size_ will be set to 0.
///   - __buffer_moved(_CharT* __ptr, size_t __capacity)
///     - This function is typically used for a dynamic sized buffer. There the
///       location of the buffer changes due to reallocations.
///     - __ptr_ will be set to __ptr. (This value may be the old value of
///       __ptr_).
///     - __capacity_ will be set to __capacity. (This value may be the old
///       value of __capacity_).
///     - __size_ remains unchanged,
///     - The range [__ptr, __ptr + __size_) contains the original data of the
///       range [__ptr_, __ptr_ + __size_).
///
/// The push_back function expects a valid buffer and a capacity of at least 1.
/// This means:
/// - The class is constructed with a valid buffer,
/// - __buffer_moved is called with a valid buffer is used before the first
///   write operation,
/// - no write function is ever called, or
/// - the class is constructed with a __max_output_size object with __max_size 0.
///
/// The latter option allows formatted_size to use the output buffer without
/// ever writing anything to the buffer.
template <class _CharT>
class __fmt_output_buffer
{
  _CharT* __ptr_;
  size_t __capacity_;
  size_t __size_{0};
  void (*__prepare_write_)(__fmt_output_buffer<_CharT>&, size_t);
  __fmt_max_output_size* __max_output_size_;

  [[nodiscard]] _CCCL_API constexpr size_t __available() const
  {
    return __capacity_ - __size_;
  }

  _CCCL_API constexpr void __prepare_write(size_t __code_units)
  {
    // Always have space for one additional code unit. This is a precondition of the push_back function.
    __code_units += 1;
    if (__available() < __code_units)
    {
      __prepare_write_(*this, __code_units + 1);
    }
  }

public:
  using value_type _CCCL_NODEBUG_ALIAS           = _CharT;
  using __prepare_write_type _CCCL_NODEBUG_ALIAS = void (*)(__fmt_output_buffer<_CharT>&, size_t);

  _CCCL_API constexpr explicit __fmt_output_buffer(
    _CharT* __ptr,
    size_t __capacity,
    __prepare_write_type __function,
    __fmt_max_output_size* __max_output_size = nullptr)
      : __ptr_{__ptr}
      , __capacity_{__capacity}
      , __prepare_write_{__function}
      , __max_output_size_{__max_output_size}
  {}

  _CCCL_API constexpr void __buffer_flushed()
  {
    __size_ = 0;
  }

  _CCCL_API constexpr void __buffer_moved(_CharT* __ptr, size_t __capacity)
  {
    __ptr_      = __ptr;
    __capacity_ = __capacity;
  }

  [[nodiscard]] _CCCL_API constexpr auto __make_output_iterator()
  {
    return __back_insert_iterator{*this};
  }

  // Used in std::back_insert_iterator.
  _CCCL_API constexpr void push_back(_CharT __c)
  {
    if (__max_output_size_ && __max_output_size_->__write_request(1) == 0)
    {
      return;
    }

    _CCCL_ASSERT(__ptr_ != nullptr, "attempted to write outside the buffer - invalid __ptr_");
    _CCCL_ASSERT(__size_ < __capacity_, "attempted to write outside the buffer - overflow");

    __ptr_[__size_++] = __c;

    // Profiling showed flushing after adding is more efficient than flushing when entering the function.
    if (__size_ == __capacity_)
    {
      __prepare_write(0);
    }
  }

  /// Copies the input __str to the buffer.
  ///
  /// Since some of the input is generated by std::to_chars, there needs to be a conversion when _CharT is wchar_t.
  template <class _InCharT>
  _CCCL_API constexpr void __copy(basic_string_view<_InCharT> __str)
  {
    // When the underlying iterator is a simple iterator the __capacity_ is
    // infinite. For a string or container back_inserter it isn't. This means
    // that adding a large string to the buffer can cause some overhead. In that
    // case a better approach could be:
    // - flush the buffer
    // - container.append(__str.begin(), __str.end());
    // The same holds true for the fill.
    // For transform it might be slightly harder, however the use case for
    // transform is slightly less common; it converts hexadecimal values to
    // upper case. For integral these strings are short.
    // TODO FMT Look at the improvements above.
    size_t __n = __str.size();
    if (__max_output_size_)
    {
      __n = __max_output_size_->__write_request(__n);
      if (__n == 0)
      {
        return;
      }
    }

    const _InCharT* __first = __str.data();
    do
    {
      __prepare_write(__n);
      size_t __chunk = ::cuda::std::min(__n, __available());
      ::cuda::std::copy_n(__first, __chunk, ::cuda::std::addressof(__ptr_[__size_]));
      __size_ += __chunk;
      __first += __chunk;
      __n -= __chunk;
    } while (__n);
  }

  /// A std::transform wrapper.
  ///
  /// Like @ref __copy it may need to do type conversion.
  template <class _Iterator, class _UnaryOperation, class _InCharT = typename iterator_traits<_Iterator>::value_type>
  _CCCL_API constexpr void __transform(_Iterator __first, _Iterator __last, _UnaryOperation __operation)
  {
    static_assert(contiguous_iterator<_Iterator>);

    _CCCL_ASSERT(__first <= __last, "not a valid range");

    auto __n = static_cast<size_t>(__last - __first);
    if (__max_output_size_)
    {
      __n = __max_output_size_->__write_request(__n);
      if (__n == 0)
      {
        return;
      }
    }

    do
    {
      __prepare_write(__n);
      size_t __chunk = ::cuda::std::min(__n, __available());
      ::cuda::std::transform(__first, __first + __chunk, ::cuda::std::addressof(__ptr_[__size_]), __operation);
      __size_ += __chunk;
      __first += __chunk;
      __n -= __chunk;
    } while (__n);
  }

  /// A \c fill_n wrapper.
  _CCCL_API constexpr void __fill(size_t __n, _CharT __value)
  {
    if (__max_output_size_)
    {
      __n = __max_output_size_->__write_request(__n);
      if (__n == 0)
      {
        return;
      }
    }

    do
    {
      __prepare_write(__n);
      size_t __chunk = ::cuda::std::min(__n, __available());
      ::cuda::std::fill_n(::cuda::std::addressof(__ptr_[__size_]), __chunk, __value);
      __size_ += __chunk;
      __n -= __chunk;
    } while (__n);
  }

  [[nodiscard]] _CCCL_API constexpr size_t __capacity() const
  {
    return __capacity_;
  }
  [[nodiscard]] _CCCL_API constexpr size_t __size() const
  {
    return __size_;
  }
};

/// Extract the container type of a \ref back_insert_iterator.
template <class _It, class = void>
struct __fmt_back_insert_iterator_container
{
  using type _CCCL_NODEBUG_ALIAS = void;
};

template <class _Container>
struct __fmt_back_insert_iterator_container<__back_insert_iterator<_Container>, enable_if_t<__fmt_insertable<_Container>>>
{
  using type _CCCL_NODEBUG_ALIAS = _Container;
};

// A dynamically growing buffer.
template <class _CharT>
class __fmt_allocating_buffer : public __fmt_output_buffer<_CharT>
{
  using _Alloc _CCCL_NODEBUG_ALIAS = allocator<_CharT>;

  // Since allocating is expensive the class has a small internal buffer. When
  // its capacity is exceeded a dynamic buffer will be allocated.
  static constexpr size_t __buffer_size = 256;
  _CharT __small_buffer_[__buffer_size];

  _CharT* __ptr_{__small_buffer_};

  _CCCL_API constexpr void __grow_buffer(size_t __capacity)
  {
    if (__capacity < __buffer_size)
    {
      return;
    }

    _CCCL_ASSERT(__capacity > this->__capacity(), "the buffer must grow");

    // _CharT is an implicit lifetime type so can be used without explicit
    // construction or destruction.
    _Alloc __alloc;
    auto __result = ::cuda::std::__allocate_at_least(__alloc, __capacity);
    ::cuda::std::copy_n(__ptr_, this->__size(), __result.ptr);
    if (__ptr_ != __small_buffer_)
    {
      __alloc.deallocate(__ptr_, this->__capacity());
    }

    __ptr_ = __result.ptr;
    this->__buffer_moved(__ptr_, __result.count);
  }

  _CCCL_API constexpr void __prepare_write(size_t __size_hint)
  {
    __grow_buffer(::cuda::std::max<size_t>(this->__capacity() + __size_hint, this->__capacity() * 1.6));
  }

  _CCCL_API static constexpr void __prepare_write(__fmt_output_buffer<_CharT>& __buffer, size_t __size_hint)
  {
    static_cast<__fmt_allocating_buffer<_CharT>&>(__buffer).__prepare_write(__size_hint);
  }

public:
  _CCCL_API constexpr explicit __fmt_allocating_buffer(__fmt_max_output_size* __max_output_size = nullptr)
      : __fmt_output_buffer<_CharT>{__small_buffer_, __buffer_size, __prepare_write, __max_output_size}
  {}

  __fmt_allocating_buffer(const __fmt_allocating_buffer&)            = delete;
  __fmt_allocating_buffer& operator=(const __fmt_allocating_buffer&) = delete;

  _CCCL_API _CCCL_CONSTEXPR_CXX20 ~__fmt_allocating_buffer()
  {
    if (__ptr_ != __small_buffer_)
    {
      _Alloc{}.deallocate(__ptr_, this->__capacity());
    }
  }

  [[nodiscard]] _CCCL_API constexpr basic_string_view<_CharT> __view()
  {
    return {__ptr_, this->__size()};
  }
};

// A buffer that directly writes to the underlying buffer.
template <class _OutIt, class _CharT>
class __fmt_direct_iterator_buffer : public __fmt_output_buffer<_CharT>
{
  // The function format_to expects a buffer large enough for the output. The
  // function format_to_n has its own helper class that restricts the number of
  // write options. So this function class can pretend to have an infinite
  // buffer.
  static constexpr size_t __buffer_size = ~0ull;

  _OutIt __out_it_;

  _CCCL_API static constexpr void
  __prepare_write([[maybe_unused]] __fmt_output_buffer<_CharT>& __buffer, [[maybe_unused]] size_t __size_hint)
  {
    _CCCL_THROW(::std::length_error, "cuda::std::__fmt_direct_iterator_buffer");
  }

public:
  _CCCL_API constexpr explicit __fmt_direct_iterator_buffer(
    _OutIt __out_it, __fmt_max_output_size* __max_output_size = nullptr)
      : __fmt_output_buffer<_CharT>{::cuda::std::__unwrap_iter(__out_it),
                                    __buffer_size,
                                    __prepare_write,
                                    __max_output_size}
      , __out_it_{__out_it}
  {}

  [[nodiscard]] _CCCL_API constexpr _OutIt __out_it() &&
  {
    return __out_it_ + this->__size();
  }
};

// A buffer that writes its output to the end of a container.
template <class _OutIt, class _CharT>
class __fmt_container_inserter_buffer : public __fmt_output_buffer<_CharT>
{
  typename __fmt_back_insert_iterator_container<_OutIt>::type* __container_;

  // This class uses a fixed size buffer and appends the elements in
  // __buffer_size chunks. An alternative would be to use an allocating buffer
  // and append the output in a single write operation. Benchmarking showed no
  // performance difference.
  static constexpr size_t __buffer_size = 256;
  _CharT __small_buffer_[__buffer_size];

  _CCCL_API constexpr void __prepare_write()
  {
    __container_->insert(__container_->end(), __small_buffer_, __small_buffer_ + this->__size());
    this->__buffer_flushed();
  }

  _CCCL_API static constexpr void
  __prepare_write(__fmt_output_buffer<_CharT>& __buffer, [[maybe_unused]] size_t __size_hint)
  {
    static_cast<__fmt_container_inserter_buffer<_OutIt, _CharT>&>(__buffer).__prepare_write();
  }

public:
  _CCCL_API constexpr explicit __fmt_container_inserter_buffer(
    _OutIt __out_it, __fmt_max_output_size* __max_output_size = nullptr)
      : __fmt_output_buffer<_CharT>{__small_buffer_, __buffer_size, __prepare_write, __max_output_size}
      , __container_{__out_it.__get_container()}
  {}

  [[nodiscard]] _CCCL_API constexpr auto __out_it() &&
  {
    __container_->insert(__container_->end(), __small_buffer_, __small_buffer_ + this->__size());
    return __back_insert_iterator{*__container_};
  }
};

// A buffer that writes to an iterator.
//
// Unlike the __container_inserter_buffer this class' performance does benefit
// from allocating and then inserting.
template <class _OutIt, class _CharT>
class __fmt_iterator_buffer : public __fmt_allocating_buffer<_CharT>
{
  _OutIt __out_it_;

public:
  _CCCL_API constexpr explicit __fmt_iterator_buffer(_OutIt __out_it, __fmt_max_output_size* __max_output_size = nullptr)
      : __fmt_allocating_buffer<_CharT>{__max_output_size}
      , __out_it_{::cuda::std::move(__out_it)}
  {}

  [[nodiscard]] _CCCL_API constexpr auto __out_it() &&
  {
    return ::cuda::std::copy(this->__view().begin(), this->__view().end(), ::cuda::std::move(__out_it_));
  }
};

// Selects the type of the buffer used for the output iterator.
template <class _OutIt, class _CharT, class _Container = typename __fmt_back_insert_iterator_container<_OutIt>::type>
using __fmt_buffer_select_t _CCCL_NODEBUG_ALIAS =
  conditional_t<!same_as<_Container, void>,
                __fmt_container_inserter_buffer<_OutIt, _CharT>,
                conditional_t<__fmt_enable_direct_output<_OutIt, _CharT>,
                              __fmt_direct_iterator_buffer<_OutIt, _CharT>,
                              __fmt_iterator_buffer<_OutIt, _CharT>>>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_BUFFER_H
