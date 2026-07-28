//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MIN_ALLOCATOR_H
#define MIN_ALLOCATOR_H

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/climits>
#include <cuda/std/cstddef>
#include <cuda/std/cstdlib>

#include "test_macros.h"

template <class T>
class bare_allocator
{
public:
  using value_type = T;

  TEST_FUNC bare_allocator() noexcept {}

  template <class U>
  TEST_FUNC bare_allocator(bare_allocator<U>) noexcept
  {}

  TEST_FUNC T* allocate(cuda::std::size_t n)
  {
    return static_cast<T*>(::operator new(n * sizeof(T)));
  }

  TEST_FUNC void deallocate(T* p, cuda::std::size_t) noexcept
  {
    return ::operator delete(static_cast<void*>(p));
  }

  TEST_FUNC friend bool operator==(bare_allocator, bare_allocator)
  {
    return true;
  }
  TEST_FUNC friend bool operator!=(bare_allocator, bare_allocator)
  {
    return false;
  }
};

template <class T>
class no_default_allocator
{
  no_default_allocator() = delete;
  struct construct_tag
  {};
  TEST_FUNC explicit no_default_allocator(construct_tag) {}

public:
  TEST_FUNC static no_default_allocator create()
  {
    construct_tag tag;
    return no_default_allocator(tag);
  }

public:
  using value_type = T;

  template <class U>
  TEST_FUNC no_default_allocator(no_default_allocator<U>) noexcept
  {}

  TEST_FUNC T* allocate(cuda::std::size_t n)
  {
    return static_cast<T*>(::operator new(n * sizeof(T)));
  }

  TEST_FUNC void deallocate(T* p, cuda::std::size_t) noexcept
  {
    return ::operator delete(static_cast<void*>(p));
  }

  TEST_FUNC friend bool operator==(no_default_allocator, no_default_allocator)
  {
    return true;
  }
  TEST_FUNC friend bool operator!=(no_default_allocator, no_default_allocator)
  {
    return false;
  }
};

TEST_GLOBAL_VARIABLE size_t malloc_allocator_base_outstanding_bytes         = 0;
TEST_GLOBAL_VARIABLE size_t malloc_allocator_base_alloc_count               = 0;
TEST_GLOBAL_VARIABLE size_t malloc_allocator_base_dealloc_count             = 0;
TEST_GLOBAL_VARIABLE bool malloc_allocator_base_disable_default_constructor = false;

struct malloc_allocator_base
{
  TEST_FUNC static size_t outstanding_alloc()
  {
    assert(malloc_allocator_base_alloc_count >= malloc_allocator_base_dealloc_count);
    return (malloc_allocator_base_alloc_count - malloc_allocator_base_dealloc_count);
  }

  TEST_FUNC static void reset()
  {
    assert(outstanding_alloc() == 0);
    malloc_allocator_base_disable_default_constructor = false;
    malloc_allocator_base_outstanding_bytes           = 0;
    malloc_allocator_base_alloc_count                 = 0;
    malloc_allocator_base_dealloc_count               = 0;
  }
};

template <class T>
class malloc_allocator : public malloc_allocator_base
{
public:
  using value_type = T;

  TEST_FUNC malloc_allocator() noexcept
  {
    assert(!malloc_allocator_base_disable_default_constructor);
  }

  template <class U>
  TEST_FUNC malloc_allocator(malloc_allocator<U>) noexcept
  {}

  TEST_FUNC T* allocate(cuda::std::size_t n)
  {
    const size_t nbytes = n * sizeof(T);
    ++malloc_allocator_base_alloc_count;
    malloc_allocator_base_outstanding_bytes += nbytes;
    return static_cast<T*>(cuda::std::malloc(nbytes));
  }

  TEST_FUNC void deallocate(T* p, cuda::std::size_t n) noexcept
  {
    const size_t nbytes = n * sizeof(T);
    ++malloc_allocator_base_dealloc_count;
    malloc_allocator_base_outstanding_bytes -= nbytes;
    cuda::std::free(static_cast<void*>(p));
  }

  TEST_FUNC friend bool operator==(malloc_allocator, malloc_allocator)
  {
    return true;
  }
  TEST_FUNC friend bool operator!=(malloc_allocator, malloc_allocator)
  {
    return false;
  }
};

TEST_GLOBAL_VARIABLE bool cpp03_allocator_construct_called = false;
template <class T>
struct cpp03_allocator : bare_allocator<T>
{
  using value_type = T;
  using pointer    = value_type*;

  // Returned value is not used but it's not prohibited.
  TEST_FUNC pointer construct(pointer p, const value_type& val)
  {
    ::new (p) value_type(val);
    cpp03_allocator_construct_called = true;
    return p;
  }

  TEST_FUNC cuda::std::size_t max_size() const
  {
    return UINT_MAX / sizeof(T);
  }
};

TEST_GLOBAL_VARIABLE bool cpp03_overload_allocator_construct_called = false;
template <class T>
struct cpp03_overload_allocator : bare_allocator<T>
{
  using value_type = T;
  using pointer    = value_type*;

  TEST_FUNC void construct(pointer p, const value_type& val)
  {
    construct(p, val, cuda::std::is_class<T>());
  }
  TEST_FUNC void construct(pointer p, const value_type& val, cuda::std::true_type)
  {
    ::new (p) value_type(val);
    cpp03_overload_allocator_construct_called = true;
  }
  TEST_FUNC void construct(pointer p, const value_type& val, cuda::std::false_type)
  {
    ::new (p) value_type(val);
    cpp03_overload_allocator_construct_called = true;
  }

  TEST_FUNC cuda::std::size_t max_size() const
  {
    return UINT_MAX / sizeof(T);
  }
};

template <class T, class = cuda::std::integral_constant<size_t, 0>>
class min_pointer;
template <class T, class ID>
class min_pointer<const T, ID>;
template <class ID>
class min_pointer<void, ID>;
template <class ID>
class min_pointer<const void, ID>;
template <class T>
class min_allocator;

template <class ID>
class min_pointer<const void, ID>
{
  const void* ptr_;

public:
  min_pointer() noexcept = default;
  TEST_FUNC min_pointer(cuda::std::nullptr_t) noexcept
      : ptr_(nullptr)
  {}
  template <class T>
  TEST_FUNC min_pointer(min_pointer<T, ID> p) noexcept
      : ptr_(p.ptr_)
  {}

  TEST_FUNC explicit operator bool() const
  {
    return ptr_ != nullptr;
  }

  TEST_FUNC friend bool operator==(min_pointer x, min_pointer y)
  {
    return x.ptr_ == y.ptr_;
  }
  TEST_FUNC friend bool operator!=(min_pointer x, min_pointer y)
  {
    return !(x == y);
  }
  template <class U, class XID>
  friend class min_pointer;
};

template <class ID>
class min_pointer<void, ID>
{
  void* ptr_;

public:
  min_pointer() noexcept = default;
  TEST_FUNC min_pointer(cuda::std::nullptr_t) noexcept
      : ptr_(nullptr)
  {}
  template <class T, class = typename cuda::std::enable_if<!cuda::std::is_const<T>::value>::type>
  TEST_FUNC min_pointer(min_pointer<T, ID> p) noexcept
      : ptr_(p.ptr_)
  {}

  TEST_FUNC explicit operator bool() const
  {
    return ptr_ != nullptr;
  }

  TEST_FUNC friend bool operator==(min_pointer x, min_pointer y)
  {
    return x.ptr_ == y.ptr_;
  }
  TEST_FUNC friend bool operator!=(min_pointer x, min_pointer y)
  {
    return !(x == y);
  }
  template <class U, class XID>
  friend class min_pointer;
};

template <class T, class ID>
class min_pointer
{
  T* ptr_;

  TEST_FUNC explicit min_pointer(T* p) noexcept
      : ptr_(p)
  {}

public:
  min_pointer() noexcept = default;
  TEST_FUNC min_pointer(cuda::std::nullptr_t) noexcept
      : ptr_(nullptr)
  {}
  TEST_FUNC explicit min_pointer(min_pointer<void, ID> p) noexcept
      : ptr_(static_cast<T*>(p.ptr_))
  {}

  TEST_FUNC explicit operator bool() const
  {
    return ptr_ != nullptr;
  }

  using difference_type   = cuda::std::ptrdiff_t;
  using reference         = T&;
  using pointer           = T*;
  using value_type        = T;
  using iterator_category = cuda::std::random_access_iterator_tag;

  TEST_FUNC reference operator*() const
  {
    return *ptr_;
  }
  TEST_FUNC pointer operator->() const
  {
    return ptr_;
  }

  TEST_FUNC min_pointer& operator++()
  {
    ++ptr_;
    return *this;
  }
  TEST_FUNC min_pointer operator++(int)
  {
    min_pointer tmp(*this);
    ++ptr_;
    return tmp;
  }

  TEST_FUNC min_pointer& operator--()
  {
    --ptr_;
    return *this;
  }
  TEST_FUNC min_pointer operator--(int)
  {
    min_pointer tmp(*this);
    --ptr_;
    return tmp;
  }

  TEST_FUNC min_pointer& operator+=(difference_type n)
  {
    ptr_ += n;
    return *this;
  }
  TEST_FUNC min_pointer& operator-=(difference_type n)
  {
    ptr_ -= n;
    return *this;
  }

  TEST_FUNC min_pointer operator+(difference_type n) const
  {
    min_pointer tmp(*this);
    tmp += n;
    return tmp;
  }

  TEST_FUNC friend min_pointer operator+(difference_type n, min_pointer x)
  {
    return x + n;
  }

  TEST_FUNC min_pointer operator-(difference_type n) const
  {
    min_pointer tmp(*this);
    tmp -= n;
    return tmp;
  }

  TEST_FUNC friend difference_type operator-(min_pointer x, min_pointer y)
  {
    return x.ptr_ - y.ptr_;
  }

  TEST_FUNC reference operator[](difference_type n) const
  {
    return ptr_[n];
  }

  TEST_FUNC friend bool operator<(min_pointer x, min_pointer y)
  {
    return x.ptr_ < y.ptr_;
  }
  TEST_FUNC friend bool operator>(min_pointer x, min_pointer y)
  {
    return y < x;
  }
  TEST_FUNC friend bool operator<=(min_pointer x, min_pointer y)
  {
    return !(y < x);
  }
  TEST_FUNC friend bool operator>=(min_pointer x, min_pointer y)
  {
    return !(x < y);
  }

  TEST_FUNC static min_pointer pointer_to(T& t)
  {
    return min_pointer(cuda::std::addressof(t));
  }

  TEST_FUNC friend bool operator==(min_pointer x, min_pointer y)
  {
    return x.ptr_ == y.ptr_;
  }
  TEST_FUNC friend bool operator!=(min_pointer x, min_pointer y)
  {
    return !(x == y);
  }
  template <class U, class XID>
  friend class min_pointer;
  template <class U>
  friend class min_allocator;
};

template <class T, class ID>
class min_pointer<const T, ID>
{
  const T* ptr_;

  TEST_FUNC explicit min_pointer(const T* p)
      : ptr_(p)
  {}

public:
  min_pointer() noexcept = default;
  TEST_FUNC min_pointer(cuda::std::nullptr_t)
      : ptr_(nullptr)
  {}
  TEST_FUNC min_pointer(min_pointer<T, ID> p)
      : ptr_(p.ptr_)
  {}
  TEST_FUNC explicit min_pointer(min_pointer<const void, ID> p)
      : ptr_(static_cast<const T*>(p.ptr_))
  {}

  TEST_FUNC explicit operator bool() const
  {
    return ptr_ != nullptr;
  }

  using difference_type   = cuda::std::ptrdiff_t;
  using reference         = const T&;
  using pointer           = const T*;
  using value_type        = const T;
  using iterator_category = cuda::std::random_access_iterator_tag;

  TEST_FUNC reference operator*() const
  {
    return *ptr_;
  }
  TEST_FUNC pointer operator->() const
  {
    return ptr_;
  }

  TEST_FUNC min_pointer& operator++()
  {
    ++ptr_;
    return *this;
  }
  TEST_FUNC min_pointer operator++(int)
  {
    min_pointer tmp(*this);
    ++ptr_;
    return tmp;
  }

  TEST_FUNC min_pointer& operator--()
  {
    --ptr_;
    return *this;
  }
  TEST_FUNC min_pointer operator--(int)
  {
    min_pointer tmp(*this);
    --ptr_;
    return tmp;
  }

  TEST_FUNC min_pointer& operator+=(difference_type n)
  {
    ptr_ += n;
    return *this;
  }
  TEST_FUNC min_pointer& operator-=(difference_type n)
  {
    ptr_ -= n;
    return *this;
  }

  TEST_FUNC min_pointer operator+(difference_type n) const
  {
    min_pointer tmp(*this);
    tmp += n;
    return tmp;
  }

  TEST_FUNC friend min_pointer operator+(difference_type n, min_pointer x)
  {
    return x + n;
  }

  TEST_FUNC min_pointer operator-(difference_type n) const
  {
    min_pointer tmp(*this);
    tmp -= n;
    return tmp;
  }

  TEST_FUNC friend difference_type operator-(min_pointer x, min_pointer y)
  {
    return x.ptr_ - y.ptr_;
  }

  TEST_FUNC reference operator[](difference_type n) const
  {
    return ptr_[n];
  }

  TEST_FUNC friend bool operator<(min_pointer x, min_pointer y)
  {
    return x.ptr_ < y.ptr_;
  }
  TEST_FUNC friend bool operator>(min_pointer x, min_pointer y)
  {
    return y < x;
  }
  TEST_FUNC friend bool operator<=(min_pointer x, min_pointer y)
  {
    return !(y < x);
  }
  TEST_FUNC friend bool operator>=(min_pointer x, min_pointer y)
  {
    return !(x < y);
  }

  TEST_FUNC static min_pointer pointer_to(const T& t)
  {
    return min_pointer(cuda::std::addressof(t));
  }

  TEST_FUNC friend bool operator==(min_pointer x, min_pointer y)
  {
    return x.ptr_ == y.ptr_;
  }
  TEST_FUNC friend bool operator!=(min_pointer x, min_pointer y)
  {
    return !(x == y);
  }
  template <class U, class XID>
  friend class min_pointer;
};

template <class T, class ID>
TEST_FUNC inline bool operator==(min_pointer<T, ID> x, cuda::std::nullptr_t)
{
  return !static_cast<bool>(x);
}

template <class T, class ID>
TEST_FUNC inline bool operator==(cuda::std::nullptr_t, min_pointer<T, ID> x)
{
  return !static_cast<bool>(x);
}

template <class T, class ID>
TEST_FUNC inline bool operator!=(min_pointer<T, ID> x, cuda::std::nullptr_t)
{
  return static_cast<bool>(x);
}

template <class T, class ID>
TEST_FUNC inline bool operator!=(cuda::std::nullptr_t, min_pointer<T, ID> x)
{
  return static_cast<bool>(x);
}

template <class T>
class min_allocator
{
public:
  using value_type = T;
  using pointer    = min_pointer<T>;

  min_allocator() = default;
  template <class U>
  TEST_FUNC min_allocator(min_allocator<U>)
  {}

  TEST_FUNC pointer allocate(cuda::std::ptrdiff_t n)
  {
    return pointer(static_cast<T*>(::operator new(n * sizeof(T))));
  }

  TEST_FUNC void deallocate(pointer p, cuda::std::ptrdiff_t) noexcept
  {
    return ::operator delete(p.ptr_);
  }

  TEST_FUNC friend bool operator==(min_allocator, min_allocator)
  {
    return true;
  }
  TEST_FUNC friend bool operator!=(min_allocator, min_allocator)
  {
    return false;
  }
};

template <class T>
class explicit_allocator
{
public:
  using value_type = T;

  TEST_FUNC explicit_allocator() noexcept {}

  template <class U>
  TEST_FUNC explicit explicit_allocator(explicit_allocator<U>) noexcept
  {}

  TEST_FUNC T* allocate(cuda::std::size_t n)
  {
    return static_cast<T*>(::operator new(n * sizeof(T)));
  }

  TEST_FUNC void deallocate(T* p, cuda::std::size_t) noexcept
  {
    return ::operator delete(static_cast<void*>(p));
  }

  TEST_FUNC friend bool operator==(explicit_allocator, explicit_allocator)
  {
    return true;
  }
  TEST_FUNC friend bool operator!=(explicit_allocator, explicit_allocator)
  {
    return false;
  }
};

#endif // MIN_ALLOCATOR_H
