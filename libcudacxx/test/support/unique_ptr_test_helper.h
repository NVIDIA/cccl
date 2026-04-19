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

#if TEST_CUDA_COMPILER(NVCC) || TEST_COMPILER(NVRTC)
TEST_NV_DIAG_SUPPRESS(3060) // call to __builtin_is_constant_evaluated appearing in a non-constexpr function
#endif // TEST_CUDA_COMPILER(NVCC) || TEST_COMPILER(NVRTC)
TEST_DIAG_SUPPRESS_GCC("-Wtautological-compare")
TEST_DIAG_SUPPRESS_CLANG("-Wtautological-compare")

TEST_GLOBAL_VARIABLE int A_count = 0;

struct A
{
  TEST_FUNC TEST_CONSTEXPR_CXX23 A()
  {
    if (!::cuda::std::__cccl_default_is_constant_evaluated())
    {
      ++A_count;
    }
  }
  TEST_FUNC TEST_CONSTEXPR_CXX23 A(const A&)
  {
    if (!::cuda::std::__cccl_default_is_constant_evaluated())
    {
      ++A_count;
    }
  }
  TEST_FUNC TEST_CONSTEXPR_CXX23 virtual ~A()
  {
    if (!::cuda::std::__cccl_default_is_constant_evaluated())
    {
      --A_count;
    }
  }
};

TEST_GLOBAL_VARIABLE int B_count = 0;

struct B : public A
{
  TEST_FUNC TEST_CONSTEXPR_CXX23 B()
  {
    if (!::cuda::std::__cccl_default_is_constant_evaluated())
    {
      ++B_count;
    }
  }
  TEST_FUNC TEST_CONSTEXPR_CXX23 B(const B&)
      : A()
  {
    if (!::cuda::std::__cccl_default_is_constant_evaluated())
    {
      ++B_count;
    }
  }
  TEST_FUNC virtual TEST_CONSTEXPR_CXX23 ~B()
  {
    if (!::cuda::std::__cccl_default_is_constant_evaluated())
    {
      --B_count;
    }
  }
};

template <class T>
typename cuda::std::enable_if<!cuda::std::is_array<T>::value, T*>::type TEST_FUNC TEST_CONSTEXPR_CXX23
newValue(int num_elements)
{
  assert(num_elements == 1);
  return new T;
}

template <class T>
typename cuda::std::enable_if<cuda::std::is_array<T>::value, typename cuda::std::remove_all_extents<T>::type*>::type
  TEST_FUNC TEST_CONSTEXPR_CXX23
  newValue(int num_elements)
{
  using VT = typename cuda::std::remove_all_extents<T>::type;
  assert(num_elements >= 1);
  return new VT[num_elements];
}

struct IncompleteType;

TEST_FUNC void checkNumIncompleteTypeAlive(int i);
TEST_FUNC int getNumIncompleteTypeAlive();
TEST_FUNC IncompleteType* getNewIncomplete();
TEST_FUNC IncompleteType* getNewIncompleteArray(int size);

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
  TEST_FUNC StoresIncomplete(Args&&... args)
      : m_ptr(cuda::std::forward<Args>(args)...)
  {
    static_assert(!args_is_this_type<StoresIncomplete, Args...>::value);
  }

  TEST_FUNC ~StoresIncomplete();

  TEST_FUNC IncompleteType* get() const
  {
    return m_ptr.get();
  }
  TEST_FUNC Del& get_deleter()
  {
    return m_ptr.get_deleter();
  }
};

template <class IncompleteT = IncompleteType, class Del = cuda::std::default_delete<IncompleteT>, class... Args>
TEST_FUNC void doIncompleteTypeTest(int expect_alive, Args&&... ctor_args)
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

#define INCOMPLETE_TEST_EPILOGUE()                                         \
  _LIBCUDACXX_DEVICE int is_incomplete_test_anchor = is_incomplete_test(); \
                                                                           \
  TEST_GLOBAL_VARIABLE int IncompleteType_count = 0;                       \
  struct IncompleteType                                                    \
  {                                                                        \
    TEST_FUNC IncompleteType()                                             \
    {                                                                      \
      ++IncompleteType_count;                                              \
    }                                                                      \
    TEST_FUNC ~IncompleteType()                                            \
    {                                                                      \
      --IncompleteType_count;                                              \
    }                                                                      \
  };                                                                       \
                                                                           \
  TEST_FUNC void checkNumIncompleteTypeAlive(int i)                        \
  {                                                                        \
    assert(IncompleteType_count == i);                                     \
  }                                                                        \
  TEST_FUNC int getNumIncompleteTypeAlive()                                \
  {                                                                        \
    return IncompleteType_count;                                           \
  }                                                                        \
  TEST_FUNC IncompleteType* getNewIncomplete()                             \
  {                                                                        \
    return new IncompleteType;                                             \
  }                                                                        \
  TEST_FUNC IncompleteType* getNewIncompleteArray(int size)                \
  {                                                                        \
    return new IncompleteType[size];                                       \
  }                                                                        \
                                                                           \
  template <class IncompleteT, class Del>                                  \
  TEST_FUNC StoresIncomplete<IncompleteT, Del>::~StoresIncomplete()        \
  {}

#define DEFINE_AND_RUN_IS_INCOMPLETE_TEST(...)        \
  TEST_FUNC static constexpr int is_incomplete_test() \
  {                                                   \
    __VA_ARGS__ return 0;                             \
  }                                                   \
  INCOMPLETE_TEST_EPILOGUE()

#endif // TEST_SUPPORT_UNIQUE_PTR_TEST_HELPER_H
