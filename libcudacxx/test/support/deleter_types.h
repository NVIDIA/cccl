//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Example move-only deleter

#ifndef SUPPORT_DELETER_TYPES_H
#define SUPPORT_DELETER_TYPES_H

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "min_allocator.h"
#include "test_macros.h"

template <class T>
class Deleter
{
  int state_;

  __host__ __device__ Deleter(const Deleter&);
  __host__ __device__ Deleter& operator=(const Deleter&);

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 Deleter(Deleter&& r)
      : state_(r.state_)
  {
    r.state_ = 0;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 Deleter& operator=(Deleter&& r)
  {
    state_   = r.state_;
    r.state_ = 0;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 Deleter()
      : state_(0)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit Deleter(int s)
      : state_(s)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~Deleter()
  {
    assert(state_ >= 0);
    state_ = -1;
  }

  template <class U>
  __host__ __device__ TEST_CONSTEXPR_CXX23
  Deleter(Deleter<U>&& d, typename cuda::std::enable_if<!cuda::std::is_same<U, T>::value>::type* = 0)
      : state_(d.state())
  {
    d.set_state(0);
  }

private:
  template <class U>
  __host__ __device__ Deleter(const Deleter<U>& d,
                              typename cuda::std::enable_if<!cuda::std::is_same<U, T>::value>::type* = 0);

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const
  {
    return state_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i)
  {
    state_ = i;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p)
  {
    delete p;
  }
};

template <class T>
class Deleter<T[]>
{
  int state_;

  __host__ __device__ Deleter(const Deleter&);
  __host__ __device__ Deleter& operator=(const Deleter&);

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 Deleter(Deleter&& r)
      : state_(r.state_)
  {
    r.state_ = 0;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 Deleter& operator=(Deleter&& r)
  {
    state_   = r.state_;
    r.state_ = 0;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 Deleter()
      : state_(0)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit Deleter(int s)
      : state_(s)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~Deleter()
  {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const
  {
    return state_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i)
  {
    state_ = i;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p)
  {
    delete[] p;
  }
};

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX23 void swap(Deleter<T>& x, Deleter<T>& y)
{
  Deleter<T> t(cuda::std::move(x));
  x = cuda::std::move(y);
  y = cuda::std::move(t);
}

template <class T>
class CDeleter
{
  int state_;

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 CDeleter()
      : state_(0)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit CDeleter(int s)
      : state_(s)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~CDeleter()
  {
    assert(state_ >= 0);
    state_ = -1;
  }

  template <class U>
  __host__ __device__ TEST_CONSTEXPR_CXX23 CDeleter(const CDeleter<U>& d)
      : state_(d.state())
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const
  {
    return state_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i)
  {
    state_ = i;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p)
  {
    delete p;
  }
};

template <class T>
class CDeleter<T[]>
{
  int state_;

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 CDeleter()
      : state_(0)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit CDeleter(int s)
      : state_(s)
  {}
  template <class U>
  __host__ __device__ TEST_CONSTEXPR_CXX23 CDeleter(const CDeleter<U>& d)
      : state_(d.state())
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX23 ~CDeleter()
  {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const
  {
    return state_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i)
  {
    state_ = i;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p)
  {
    delete[] p;
  }
};

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX23 void swap(CDeleter<T>& x, CDeleter<T>& y)
{
  CDeleter<T> t(cuda::std::move(x));
  x = cuda::std::move(y);
  y = cuda::std::move(t);
}

// Non-copyable deleter
template <class T>
class NCDeleter
{
  int state_;
  __host__ __device__ NCDeleter(NCDeleter const&);
  __host__ __device__ NCDeleter& operator=(NCDeleter const&);

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 NCDeleter()
      : state_(0)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit NCDeleter(int s)
      : state_(s)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~NCDeleter()
  {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const
  {
    return state_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i)
  {
    state_ = i;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p)
  {
    delete p;
  }
};

template <class T>
class NCDeleter<T[]>
{
  int state_;
  __host__ __device__ NCDeleter(NCDeleter const&);
  __host__ __device__ NCDeleter& operator=(NCDeleter const&);

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 NCDeleter()
      : state_(0)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit NCDeleter(int s)
      : state_(s)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~NCDeleter()
  {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const
  {
    return state_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i)
  {
    state_ = i;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p)
  {
    delete[] p;
  }
};

// Non-copyable deleter
template <class T>
class NCConstDeleter
{
  int state_;
  __host__ __device__ NCConstDeleter(NCConstDeleter const&);
  __host__ __device__ NCConstDeleter& operator=(NCConstDeleter const&);

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 NCConstDeleter()
      : state_(0)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit NCConstDeleter(int s)
      : state_(s)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~NCConstDeleter()
  {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const
  {
    return state_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i)
  {
    state_ = i;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p) const
  {
    delete p;
  }
};

template <class T>
class NCConstDeleter<T[]>
{
  int state_;
  __host__ __device__ NCConstDeleter(NCConstDeleter const&);
  __host__ __device__ NCConstDeleter& operator=(NCConstDeleter const&);

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 NCConstDeleter()
      : state_(0)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit NCConstDeleter(int s)
      : state_(s)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~NCConstDeleter()
  {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const
  {
    return state_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i)
  {
    state_ = i;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p) const
  {
    delete[] p;
  }
};

// Non-copyable deleter
template <class T>
class CopyDeleter
{
  int state_;

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 CopyDeleter()
      : state_(0)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit CopyDeleter(int s)
      : state_(s)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~CopyDeleter()
  {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 CopyDeleter(CopyDeleter const& other)
      : state_(other.state_)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 CopyDeleter& operator=(CopyDeleter const& other)
  {
    state_ = other.state_;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const
  {
    return state_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i)
  {
    state_ = i;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p)
  {
    delete p;
  }
};

template <class T>
class CopyDeleter<T[]>
{
  int state_;

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 CopyDeleter()
      : state_(0)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit CopyDeleter(int s)
      : state_(s)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~CopyDeleter()
  {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 CopyDeleter(CopyDeleter const& other)
      : state_(other.state_)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 CopyDeleter& operator=(CopyDeleter const& other)
  {
    state_ = other.state_;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const
  {
    return state_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i)
  {
    state_ = i;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p)
  {
    delete[] p;
  }
};

STATIC_TEST_GLOBAL_VAR int test_deleter_base_count         = 0;
STATIC_TEST_GLOBAL_VAR int test_deleter_base_dealloc_count = 0;
struct test_deleter_base
{};

template <class T>
class test_deleter : public test_deleter_base
{
  int state_;

public:
  __host__ __device__ test_deleter()
      : state_(0)
  {
    ++test_deleter_base_count;
  }
  __host__ __device__ explicit test_deleter(int s)
      : state_(s)
  {
    ++test_deleter_base_count;
  }
  __host__ __device__ test_deleter(const test_deleter& d)
      : state_(d.state_)
  {
    ++test_deleter_base_count;
  }
  __host__ __device__ ~test_deleter()
  {
    assert(state_ >= 0);
    --test_deleter_base_count;
    state_ = -1;
  }

  __host__ __device__ int state() const
  {
    return state_;
  }
  __host__ __device__ void set_state(int i)
  {
    state_ = i;
  }

  __host__ __device__ void operator()(T* p)
  {
    assert(state_ >= 0);
    ++test_deleter_base_count;
    delete p;
  }
  __host__ __device__ test_deleter* operator&() const = delete;
};

template <class T>
__host__ __device__ void swap(test_deleter<T>& x, test_deleter<T>& y)
{
  test_deleter<T> t(cuda::std::move(x));
  x = cuda::std::move(y);
  y = cuda::std::move(t);
}

template <class T, cuda::std::size_t ID = 0>
class PointerDeleter
{
  __host__ __device__ PointerDeleter(const PointerDeleter&);
  __host__ __device__ PointerDeleter& operator=(const PointerDeleter&);

public:
  typedef min_pointer<T, cuda::std::integral_constant<cuda::std::size_t, ID>> pointer;

  TEST_CONSTEXPR_CXX23 PointerDeleter()                            = default;
  TEST_CONSTEXPR_CXX23 PointerDeleter(PointerDeleter&&)            = default;
  TEST_CONSTEXPR_CXX23 PointerDeleter& operator=(PointerDeleter&&) = default;
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit PointerDeleter(int) {}

  template <class U>
  __host__ __device__ TEST_CONSTEXPR_CXX23
  PointerDeleter(PointerDeleter<U, ID>&&, typename cuda::std::enable_if<!cuda::std::is_same<U, T>::value>::type* = 0)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(pointer p)
  {
    if (p)
    {
      delete cuda::std::addressof(*p);
    }
  }

private:
  template <class U>
  __host__ __device__ PointerDeleter(const PointerDeleter<U, ID>&,
                                     typename cuda::std::enable_if<!cuda::std::is_same<U, T>::value>::type* = 0);
};

template <class T, cuda::std::size_t ID>
class PointerDeleter<T[], ID>
{
  __host__ __device__ PointerDeleter(const PointerDeleter&);
  __host__ __device__ PointerDeleter& operator=(const PointerDeleter&);

public:
  typedef min_pointer<T, cuda::std::integral_constant<cuda::std::size_t, ID>> pointer;

  TEST_CONSTEXPR_CXX23 PointerDeleter()                            = default;
  TEST_CONSTEXPR_CXX23 PointerDeleter(PointerDeleter&&)            = default;
  TEST_CONSTEXPR_CXX23 PointerDeleter& operator=(PointerDeleter&&) = default;
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit PointerDeleter(int) {}

  template <class U>
  __host__ __device__ TEST_CONSTEXPR_CXX23
  PointerDeleter(PointerDeleter<U, ID>&&, typename cuda::std::enable_if<!cuda::std::is_same<U, T>::value>::type* = 0)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(pointer p)
  {
    if (p)
    {
      delete[] cuda::std::addressof(*p);
    }
  }

private:
  template <class U>
  __host__ __device__ PointerDeleter(const PointerDeleter<U, ID>&,
                                     typename cuda::std::enable_if<!cuda::std::is_same<U, T>::value>::type* = 0);
};

template <class T>
class DefaultCtorDeleter
{
  int state_;

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const
  {
    return state_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p)
  {
    delete p;
  }
};

template <class T>
class DefaultCtorDeleter<T[]>
{
  int state_;

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const
  {
    return state_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p)
  {
    delete[] p;
  }
};

#endif // SUPPORT_DELETER_TYPES_H
