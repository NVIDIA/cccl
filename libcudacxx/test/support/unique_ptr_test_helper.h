//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_UNIQUE_PTR_TEST_HELPER_H
#define TEST_SUPPORT_UNIQUE_PTR_TEST_HELPER_H

#include <cuda/std/__memory_>
#include <cuda/std/type_traits>

#include "deleter_types.h"
#include "test_macros.h"

#if defined(TEST_COMPILER_NVCC) || defined(TEST_COMPILER_NVRTC)
TEST_NV_DIAG_SUPPRESS(3060) // call to __builtin_is_constant_evaluated appearing in a non-constexpr function
#endif // TEST_COMPILER_NVCC || TEST_COMPILER_NVRTC
#if defined(TEST_COMPILER_GCC)
#  pragma GCC diagnostic ignored "-Wtautological-compare"
#elif defined(TEST_COMPILER_CLANG)
#  pragma clang diagnostic ignored "-Wtautological-compare"
#endif

STATIC_TEST_GLOBAL_VAR int A_count = 0;

struct A
{
  __host__ __device__ A()
  {
    ++A_count;
  }
  __host__ __device__ A(const A&)
  {
    ++A_count;
  }
  __host__ __device__ virtual ~A()
  {
    --A_count;
  }
};

STATIC_TEST_GLOBAL_VAR int B_count = 0;

struct B : public A
{
  __host__ __device__ B()
  {
    ++B_count;
  }
  __host__ __device__ B(const B&)
      : A()
  {
    ++B_count;
  }
  __host__ __device__ virtual ~B()
  {
    --B_count;
  }
};

template <class T>
typename cuda::std::enable_if<!cuda::std::is_array<T>::value, T*>::type __host__ __device__ newValue(int num_elements)
{
  assert(num_elements == 1);
  return new T;
}

template <class T>
typename cuda::std::enable_if<cuda::std::is_array<T>::value, typename cuda::std::remove_all_extents<T>::type*>::type
  __host__ __device__
  newValue(int num_elements)
{
  typedef typename cuda::std::remove_all_extents<T>::type VT;
  assert(num_elements >= 1);
  return new VT[num_elements];
}

struct IncompleteType;

__host__ __device__ void checkNumIncompleteTypeAlive(int i);
__host__ __device__ int getNumIncompleteTypeAlive();
__host__ __device__ IncompleteType* getNewIncomplete();
__host__ __device__ IncompleteType* getNewIncompleteArray(int size);

template <class ThisT, class... Args>
struct args_is_this_type : cuda::std::false_type
{};

template <class ThisT, class A1>
struct args_is_this_type<ThisT, A1> : cuda::std::is_same<ThisT, typename cuda::std::decay<A1>::type>
{};

template <class IncompleteT = IncompleteType, class Del = cuda::std::default_delete<IncompleteT>>
struct StoresIncomplete
{
  static_assert((cuda::std::is_same<IncompleteT, IncompleteType>::value
                 || cuda::std::is_same<IncompleteT, IncompleteType[]>::value),
                "");

  cuda::std::unique_ptr<IncompleteT, Del> m_ptr;

  StoresIncomplete(StoresIncomplete const&) = delete;
  StoresIncomplete(StoresIncomplete&&)      = default;

  template <class... Args>
  __host__ __device__ StoresIncomplete(Args&&... args)
      : m_ptr(cuda::std::forward<Args>(args)...)
  {
    static_assert(!args_is_this_type<StoresIncomplete, Args...>::value, "");
  }

  __host__ __device__ ~StoresIncomplete();

  __host__ __device__ IncompleteType* get() const
  {
    return m_ptr.get();
  }
  __host__ __device__ Del& get_deleter()
  {
    return m_ptr.get_deleter();
  }
};

template <class IncompleteT = IncompleteType, class Del = cuda::std::default_delete<IncompleteT>, class... Args>
__host__ __device__ void doIncompleteTypeTest(int expect_alive, Args&&... ctor_args)
{
  checkNumIncompleteTypeAlive(expect_alive);
  {
    StoresIncomplete<IncompleteT, Del> sptr(cuda::std::forward<Args>(ctor_args)...);
    checkNumIncompleteTypeAlive(expect_alive);
    if (expect_alive == 0)
    {
      assert(sptr.get() == nullptr);
    }
    else
    {
      assert(sptr.get() != nullptr);
    }
  }
  checkNumIncompleteTypeAlive(0);
}

#define INCOMPLETE_TEST_EPILOGUE()                                            \
  _LIBCUDACXX_DEVICE int is_incomplete_test_anchor = is_incomplete_test();    \
                                                                              \
  STATIC_TEST_GLOBAL_VAR int IncompleteType_count = 0;                        \
  struct IncompleteType                                                       \
  {                                                                           \
    __host__ __device__ IncompleteType()                                      \
    {                                                                         \
      ++IncompleteType_count;                                                 \
    }                                                                         \
    __host__ __device__ ~IncompleteType()                                     \
    {                                                                         \
      --IncompleteType_count;                                                 \
    }                                                                         \
  };                                                                          \
                                                                              \
  __host__ __device__ void checkNumIncompleteTypeAlive(int i)                 \
  {                                                                           \
    assert(IncompleteType_count == i);                                        \
  }                                                                           \
  __host__ __device__ int getNumIncompleteTypeAlive()                         \
  {                                                                           \
    return IncompleteType_count;                                              \
  }                                                                           \
  __host__ __device__ IncompleteType* getNewIncomplete()                      \
  {                                                                           \
    return new IncompleteType;                                                \
  }                                                                           \
  __host__ __device__ IncompleteType* getNewIncompleteArray(int size)         \
  {                                                                           \
    return new IncompleteType[size];                                          \
  }                                                                           \
                                                                              \
  template <class IncompleteT, class Del>                                     \
  __host__ __device__ StoresIncomplete<IncompleteT, Del>::~StoresIncomplete() \
  {}

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wvariadic-macros"
#endif

#define DEFINE_AND_RUN_IS_INCOMPLETE_TEST(...)                  \
  __host__ __device__ static constexpr int is_incomplete_test() \
  {                                                             \
    __VA_ARGS__ return 0;                                       \
  }                                                             \
  INCOMPLETE_TEST_EPILOGUE()

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

#endif // TEST_SUPPORT_UNIQUE_PTR_TEST_HELPER_H
