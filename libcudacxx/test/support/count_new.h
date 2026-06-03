//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COUNT_NEW_H
#define COUNT_NEW_H

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/cassert>
#include <cuda/std/cstdlib>

#if _CCCL_HOSTED()
#  include <new>
#endif // _CCCL_HOSTED()

#include "test_macros.h"

#if defined(TEST_HAS_SANITIZERS) || defined(__CUDA_ARCH__) // we cannot overload operator {new, delete} on device
#  define DISABLE_NEW_COUNT
#endif

// All checks return true when disable_checking is enabled.
#ifdef DISABLE_NEW_COUNT
TEST_GLOBAL_VARIABLE const bool MemCounter_disable_checking = true;
#else
TEST_GLOBAL_VARIABLE const bool MemCounter_disable_checking = false;
#endif

// number of allocations to throw after. Default (unsigned)-1. If
// throw_after has the default value it will never be decremented.
TEST_GLOBAL_VARIABLE const unsigned MemCounter_never_throw_value = static_cast<unsigned>(-1);

class MemCounter
{
public:
  MemCounter() = default;

  // Make MemCounter super hard to accidentally construct or copy.
  class MemCounterCtorArg_
  {};
  TEST_FUNC explicit MemCounter(MemCounterCtorArg_) {}

private:
  TEST_FUNC MemCounter(MemCounter const&)            = delete;
  TEST_FUNC MemCounter& operator=(MemCounter const&) = delete;

public:
  // Disallow any allocations from occurring. Useful for testing that
  // code doesn't perform any allocations.
  bool disable_allocations = false;

  unsigned throw_after = MemCounter_never_throw_value;

  int outstanding_new                 = 0;
  int new_called                      = 0;
  int delete_called                   = 0;
  int aligned_new_called              = 0;
  int aligned_delete_called           = 0;
  cuda::std::size_t last_new_size     = 0;
  cuda::std::size_t last_new_align    = 0;
  cuda::std::size_t last_delete_align = 0;

  int outstanding_array_new                 = 0;
  int new_array_called                      = 0;
  int delete_array_called                   = 0;
  int aligned_new_array_called              = 0;
  int aligned_delete_array_called           = 0;
  cuda::std::size_t last_new_array_size     = 0;
  cuda::std::size_t last_new_array_align    = 0;
  cuda::std::size_t last_delete_array_align = 0;

public:
  TEST_FUNC void newCalled(cuda::std::size_t s)
  {
    assert(disable_allocations == false);
    assert(s);
    if (throw_after == 0)
    {
      throw_after = MemCounter_never_throw_value;
      _CCCL_THROW(::std::bad_alloc);
    }
    else if (throw_after != MemCounter_never_throw_value)
    {
      --throw_after;
    }
    ++new_called;
    ++outstanding_new;
    last_new_size = s;
  }

  TEST_FUNC void alignedNewCalled(cuda::std::size_t s, cuda::std::size_t a)
  {
    newCalled(s);
    ++aligned_new_called;
    last_new_align = a;
  }

  TEST_FUNC void deleteCalled(void* p)
  {
    assert(p);
    --outstanding_new;
    ++delete_called;
  }

  TEST_FUNC void alignedDeleteCalled(void* p, cuda::std::size_t a)
  {
    deleteCalled(p);
    ++aligned_delete_called;
    last_delete_align = a;
  }

  TEST_FUNC void newArrayCalled(cuda::std::size_t s)
  {
    assert(disable_allocations == false);
    assert(s);
    if (throw_after == 0)
    {
      throw_after = MemCounter_never_throw_value;
      _CCCL_THROW(::std::bad_alloc);
    }
    else
    {
      // don't decrement throw_after here. newCalled will end up doing that.
    }
    ++outstanding_array_new;
    ++new_array_called;
    last_new_array_size = s;
  }

  TEST_FUNC void alignedNewArrayCalled(cuda::std::size_t s, cuda::std::size_t a)
  {
    newArrayCalled(s);
    ++aligned_new_array_called;
    last_new_array_align = a;
  }

  TEST_FUNC void deleteArrayCalled(void* p)
  {
    assert(p);
    --outstanding_array_new;
    ++delete_array_called;
  }

  TEST_FUNC void alignedDeleteArrayCalled(void* p, cuda::std::size_t a)
  {
    deleteArrayCalled(p);
    ++aligned_delete_array_called;
    last_delete_array_align = a;
  }

  TEST_FUNC void disableAllocations()
  {
    disable_allocations = true;
  }

  TEST_FUNC void enableAllocations()
  {
    disable_allocations = false;
  }

  TEST_FUNC void reset()
  {
    disable_allocations = false;
    throw_after         = MemCounter_never_throw_value;

    outstanding_new       = 0;
    new_called            = 0;
    delete_called         = 0;
    aligned_new_called    = 0;
    aligned_delete_called = 0;
    last_new_size         = 0;
    last_new_align        = 0;

    outstanding_array_new       = 0;
    new_array_called            = 0;
    delete_array_called         = 0;
    aligned_new_array_called    = 0;
    aligned_delete_array_called = 0;
    last_new_array_size         = 0;
    last_new_array_align        = 0;
  }

public:
  TEST_FUNC bool checkOutstandingNewEq(int n) const
  {
    return MemCounter_disable_checking || n == outstanding_new;
  }

  TEST_FUNC bool checkOutstandingNewNotEq(int n) const
  {
    return MemCounter_disable_checking || n != outstanding_new;
  }

  TEST_FUNC bool checkNewCalledEq(int n) const
  {
    return MemCounter_disable_checking || n == new_called;
  }

  TEST_FUNC bool checkNewCalledNotEq(int n) const
  {
    return MemCounter_disable_checking || n != new_called;
  }

  TEST_FUNC bool checkNewCalledGreaterThan(int n) const
  {
    return MemCounter_disable_checking || new_called > n;
  }

  TEST_FUNC bool checkDeleteCalledEq(int n) const
  {
    return MemCounter_disable_checking || n == delete_called;
  }

  TEST_FUNC bool checkDeleteCalledNotEq(int n) const
  {
    return MemCounter_disable_checking || n != delete_called;
  }

  TEST_FUNC bool checkAlignedNewCalledEq(int n) const
  {
    return MemCounter_disable_checking || n == aligned_new_called;
  }

  TEST_FUNC bool checkAlignedNewCalledNotEq(int n) const
  {
    return MemCounter_disable_checking || n != aligned_new_called;
  }

  TEST_FUNC bool checkAlignedNewCalledGreaterThan(int n) const
  {
    return MemCounter_disable_checking || aligned_new_called > n;
  }

  TEST_FUNC bool checkAlignedDeleteCalledEq(int n) const
  {
    return MemCounter_disable_checking || n == aligned_delete_called;
  }

  TEST_FUNC bool checkAlignedDeleteCalledNotEq(int n) const
  {
    return MemCounter_disable_checking || n != aligned_delete_called;
  }

  TEST_FUNC bool checkLastNewSizeEq(cuda::std::size_t n) const
  {
    return MemCounter_disable_checking || n == last_new_size;
  }

  TEST_FUNC bool checkLastNewSizeNotEq(cuda::std::size_t n) const
  {
    return MemCounter_disable_checking || n != last_new_size;
  }

  TEST_FUNC bool checkLastNewAlignEq(cuda::std::size_t n) const
  {
    return MemCounter_disable_checking || n == last_new_align;
  }

  TEST_FUNC bool checkLastNewAlignNotEq(cuda::std::size_t n) const
  {
    return MemCounter_disable_checking || n != last_new_align;
  }

  TEST_FUNC bool checkLastDeleteAlignEq(cuda::std::size_t n) const
  {
    return MemCounter_disable_checking || n == last_delete_align;
  }

  TEST_FUNC bool checkLastDeleteAlignNotEq(cuda::std::size_t n) const
  {
    return MemCounter_disable_checking || n != last_delete_align;
  }

  TEST_FUNC bool checkOutstandingArrayNewEq(int n) const
  {
    return MemCounter_disable_checking || n == outstanding_array_new;
  }

  TEST_FUNC bool checkOutstandingArrayNewNotEq(int n) const
  {
    return MemCounter_disable_checking || n != outstanding_array_new;
  }

  TEST_FUNC bool checkNewArrayCalledEq(int n) const
  {
    return MemCounter_disable_checking || n == new_array_called;
  }

  TEST_FUNC bool checkNewArrayCalledNotEq(int n) const
  {
    return MemCounter_disable_checking || n != new_array_called;
  }

  TEST_FUNC bool checkDeleteArrayCalledEq(int n) const
  {
    return MemCounter_disable_checking || n == delete_array_called;
  }

  TEST_FUNC bool checkDeleteArrayCalledNotEq(int n) const
  {
    return MemCounter_disable_checking || n != delete_array_called;
  }

  TEST_FUNC bool checkAlignedNewArrayCalledEq(int n) const
  {
    return MemCounter_disable_checking || n == aligned_new_array_called;
  }

  TEST_FUNC bool checkAlignedNewArrayCalledNotEq(int n) const
  {
    return MemCounter_disable_checking || n != aligned_new_array_called;
  }

  TEST_FUNC bool checkAlignedNewArrayCalledGreaterThan(int n) const
  {
    return MemCounter_disable_checking || aligned_new_array_called > n;
  }

  TEST_FUNC bool checkAlignedDeleteArrayCalledEq(int n) const
  {
    return MemCounter_disable_checking || n == aligned_delete_array_called;
  }

  TEST_FUNC bool checkAlignedDeleteArrayCalledNotEq(int n) const
  {
    return MemCounter_disable_checking || n != aligned_delete_array_called;
  }

  TEST_FUNC bool checkLastNewArraySizeEq(cuda::std::size_t n) const
  {
    return MemCounter_disable_checking || n == last_new_array_size;
  }

  TEST_FUNC bool checkLastNewArraySizeNotEq(cuda::std::size_t n) const
  {
    return MemCounter_disable_checking || n != last_new_array_size;
  }

  TEST_FUNC bool checkLastNewArrayAlignEq(cuda::std::size_t n) const
  {
    return MemCounter_disable_checking || n == last_new_array_align;
  }

  TEST_FUNC bool checkLastNewArrayAlignNotEq(cuda::std::size_t n) const
  {
    return MemCounter_disable_checking || n != last_new_array_align;
  }
};

TEST_GLOBAL_VARIABLE MemCounter counter{};

TEST_FUNC inline constexpr MemCounter* getGlobalMemCounter()
{
  return &counter;
}

TEST_GLOBAL_VARIABLE MemCounter& globalMemCounter = *getGlobalMemCounter();

#ifndef DISABLE_NEW_COUNT
void* operator new(cuda::std::size_t s)
{
  getGlobalMemCounter()->newCalled(s);
  void* ret = malloc(s);
  if (ret == nullptr)
  {
    _CCCL_THROW(::std::bad_alloc);
  }
  return ret;
}

void operator delete(void* p) noexcept
{
  getGlobalMemCounter()->deleteCalled(p);
  free(p);
}

#  if TEST_COMPILER(GCC)
void operator delete(void* p, cuda::std::size_t) noexcept
{
  getGlobalMemCounter()->deleteCalled(p);
  free(p);
}
#  endif // TEST_COMPILER(GCC)

void* operator new[](cuda::std::size_t s)
{
  getGlobalMemCounter()->newArrayCalled(s);
  return operator new(s);
}

void operator delete[](void* p) noexcept
{
  getGlobalMemCounter()->deleteArrayCalled(p);
  operator delete(p);
}

#  if TEST_COMPILER(GCC)
void operator delete[](void* p, cuda::std::size_t) noexcept
{
  getGlobalMemCounter()->deleteArrayCalled(p);
  operator delete(p);
}
#  endif // TEST_COMPILER(GCC)

#  if _LIBCUDACXX_HAS_ALIGNED_ALLOCATION()
#    if defined(_WIN32)
#      define USE_ALIGNED_ALLOC
#    endif

void* operator new(cuda::std::size_t s, cuda::std::align_val_t av)
{
  const cuda::std::size_t a = static_cast<cuda::std::size_t>(av);
  getGlobalMemCounter()->alignedNewCalled(s, a);
  void* ret;
#    ifdef USE_ALIGNED_ALLOC
  ret = _aligned_malloc(s, a);
#    else
  posix_memalign(&ret, a, s);
#    endif
  if (ret == nullptr)
  {
    _CCCL_THROW(::std::bad_alloc);
  }
  return ret;
}

void operator delete(void* p, cuda::std::align_val_t av) noexcept
{
  const cuda::std::size_t a = static_cast<cuda::std::size_t>(av);
  getGlobalMemCounter()->alignedDeleteCalled(p, a);
  if (p)
  {
#    ifdef USE_ALIGNED_ALLOC
    ::_aligned_free(p);
#    else
    ::free(p);
#    endif
  }
}

void* operator new[](cuda::std::size_t s, cuda::std::align_val_t av)
{
  const cuda::std::size_t a = static_cast<cuda::std::size_t>(av);
  getGlobalMemCounter()->alignedNewArrayCalled(s, a);
  return operator new(s, av);
}

void operator delete[](void* p, cuda::std::align_val_t av) noexcept
{
  const cuda::std::size_t a = static_cast<cuda::std::size_t>(av);
  getGlobalMemCounter()->alignedDeleteArrayCalled(p, a);
  return operator delete(p, av);
}

#  endif // _LIBCUDACXX_HAS_ALIGNED_ALLOCATION()

#endif // DISABLE_NEW_COUNT

struct DisableAllocationGuard
{
  TEST_FUNC explicit DisableAllocationGuard(bool disable = true)
      : m_disabled(disable)
  {
    // Don't re-disable if already disabled.
    if (globalMemCounter.disable_allocations == true)
    {
      m_disabled = false;
    }
    if (m_disabled)
    {
      globalMemCounter.disableAllocations();
    }
  }

  TEST_FUNC void release()
  {
    if (m_disabled)
    {
      globalMemCounter.enableAllocations();
    }
    m_disabled = false;
  }

  TEST_FUNC ~DisableAllocationGuard()
  {
    release();
  }

private:
  bool m_disabled;

  TEST_FUNC DisableAllocationGuard(DisableAllocationGuard const&)            = delete;
  TEST_FUNC DisableAllocationGuard& operator=(DisableAllocationGuard const&) = delete;
};

struct RequireAllocationGuard
{
  TEST_FUNC explicit RequireAllocationGuard(cuda::std::size_t RequireAtLeast = 1)
      : m_req_alloc(RequireAtLeast)
      , m_new_count_on_init(globalMemCounter.new_called)
      , m_outstanding_new_on_init(globalMemCounter.outstanding_new)
      , m_exactly(false)
  {}

  TEST_FUNC void requireAtLeast(cuda::std::size_t N)
  {
    m_req_alloc = N;
    m_exactly   = false;
  }
  TEST_FUNC void requireExactly(cuda::std::size_t N)
  {
    m_req_alloc = N;
    m_exactly   = true;
  }

  TEST_FUNC ~RequireAllocationGuard()
  {
    assert(globalMemCounter.checkOutstandingNewEq(static_cast<int>(m_outstanding_new_on_init)));
    cuda::std::size_t Expect = m_new_count_on_init + m_req_alloc;
    assert(globalMemCounter.checkNewCalledEq(static_cast<int>(Expect))
           || (!m_exactly && globalMemCounter.checkNewCalledGreaterThan(static_cast<int>(Expect))));
  }

private:
  cuda::std::size_t m_req_alloc;
  const cuda::std::size_t m_new_count_on_init;
  const cuda::std::size_t m_outstanding_new_on_init;
  bool m_exactly;
  TEST_FUNC RequireAllocationGuard(RequireAllocationGuard const&)            = delete;
  TEST_FUNC RequireAllocationGuard& operator=(RequireAllocationGuard const&) = delete;
};

#endif /* COUNT_NEW_H */
