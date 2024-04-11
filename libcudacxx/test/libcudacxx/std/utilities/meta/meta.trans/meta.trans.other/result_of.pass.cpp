//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// result_of<Fn(ArgTypes...)>

#define _LIBCUDACXX_ENABLE_CXX20_REMOVED_TYPE_TRAITS
#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/type_traits>
#ifdef _LIBCUDACXX_HAS_MEMORY
#  include <cuda/std/memory>
#endif // _LIBCUDACXX_HAS_MEMORY
#include <cuda/functional>
#include <cuda/std/cassert>

#include "test_macros.h"

struct S
{
  typedef short (*FreeFunc)(long);
  __host__ __device__ operator FreeFunc() const;
  __host__ __device__ double operator()(char, int&);
  __host__ __device__ double const& operator()(char, int&) const;
  __host__ __device__ double volatile& operator()(char, int&) volatile;
  __host__ __device__ double const volatile& operator()(char, int&) const volatile;
};

struct SD : public S
{};

struct NotDerived
{};

template <class Tp>
struct Voider
{
  typedef void type;
};

template <class T, class = void>
struct HasType : cuda::std::false_type
{};

template <class T>
struct HasType<T, typename Voider<typename T::type>::type> : cuda::std::true_type
{};

#if TEST_STD_VER > 2011
template <typename T, typename U>
struct test_invoke_result;

template <typename Fn, typename... Args, typename Ret>
struct test_invoke_result<Fn(Args...), Ret>
{
  __host__ __device__ static void call()
  {
    static_assert(cuda::std::is_invocable<Fn, Args...>::value, "");
    static_assert(cuda::std::is_invocable_r<Ret, Fn, Args...>::value, "");
    ASSERT_SAME_TYPE(Ret, typename cuda::std::invoke_result<Fn, Args...>::type);
  }
};
#endif

template <class T, class U>
__host__ __device__ void test_result_of()
{
  ASSERT_SAME_TYPE(U, typename cuda::std::result_of<T>::type);
#if TEST_STD_VER > 2011
  test_invoke_result<T, U>::call();
#endif
}

#if TEST_STD_VER > 2011
template <typename T>
struct test_invoke_no_result;

template <typename Fn, typename... Args>
struct test_invoke_no_result<Fn(Args...)>
{
  __host__ __device__ static void call()
  {
    static_assert(cuda::std::is_invocable<Fn, Args...>::value == false, "");
    static_assert((!HasType<cuda::std::invoke_result<Fn, Args...>>::value), "");
  }
};
#endif

template <class T>
__host__ __device__ void test_no_result()
{
  static_assert((!HasType<cuda::std::result_of<T>>::value), "");
#if TEST_STD_VER > 2011
  test_invoke_no_result<T>::call();
#endif
}

#if defined(__NVCC__)
template <class Ret, class Fn>
__host__ __device__ void test_lambda(Fn&&)
{
  ASSERT_SAME_TYPE(Ret, typename cuda::std::result_of<Fn()>::type);

#  if TEST_STD_VER > 2011
  ASSERT_SAME_TYPE(Ret, typename cuda::std::invoke_result<Fn>::type);
#  endif
}
#endif

int main(int, char**)
{
  typedef NotDerived ND;
  { // functor object
    test_result_of<S(int), short>();
    test_result_of<S&(unsigned char, int&), double>();
    test_result_of<S const&(unsigned char, int&), double const&>();
    test_result_of<S volatile&(unsigned char, int&), double volatile&>();
    test_result_of<S const volatile&(unsigned char, int&), double const volatile&>();
  }
  { // pointer to function
    typedef bool (&RF0)();
    typedef bool* (&RF1)(int);
    typedef bool& (&RF2)(int, int);
    typedef bool const& (&RF3)(int, int, int);
    typedef bool (&RF4)(int, ...);
    typedef bool (*PF0)();
    typedef bool* (*PF1)(int);
    typedef bool& (*PF2)(int, int);
    typedef bool const& (*PF3)(int, int, int);
    typedef bool (*PF4)(int, ...);
    typedef bool (*&PRF0)();
    typedef bool* (*&PRF1)(int);
    typedef bool& (*&PRF2)(int, int);
    typedef bool const& (*&PRF3)(int, int, int);
    typedef bool (*&PRF4)(int, ...);
    test_result_of<RF0(), bool>();
    test_result_of<RF1(int), bool*>();
    test_result_of<RF2(int, long), bool&>();
    test_result_of<RF3(int, long, int), bool const&>();
    test_result_of<RF4(int, float, void*), bool>();
    test_result_of<PF0(), bool>();
    test_result_of<PF1(int), bool*>();
    test_result_of<PF2(int, long), bool&>();
    test_result_of<PF3(int, long, int), bool const&>();
    test_result_of<PF4(int, float, void*), bool>();
    test_result_of<PRF0(), bool>();
    test_result_of<PRF1(int), bool*>();
    test_result_of<PRF2(int, long), bool&>();
    test_result_of<PRF3(int, long, int), bool const&>();
    test_result_of<PRF4(int, float, void*), bool>();
  }
  { // pointer to member function

    typedef int (S::*PMS0)();
    typedef int* (S::*PMS1)(long);
    typedef int& (S::*PMS2)(long, int);
    typedef const int& (S::*PMS3)(int, ...);
    test_result_of<PMS0(S), int>();
    test_result_of<PMS0(S&), int>();
    test_result_of<PMS0(S*), int>();
    test_result_of<PMS0(S*&), int>();

    test_result_of<PMS0(cuda::std::reference_wrapper<S>), int>();
    test_result_of<PMS0(const cuda::std::reference_wrapper<S>&), int>();
    test_result_of<PMS0(cuda::std::reference_wrapper<SD>), int>();
    test_result_of<PMS0(const cuda::std::reference_wrapper<SD>&), int>();

#ifdef _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMS0(cuda::std::unique_ptr<S>), int>();
    test_result_of<PMS0(cuda::std::unique_ptr<SD>), int>();
#endif // _LIBCUDACXX_HAS_MEMORY
    test_no_result<PMS0(const S&)>();
    test_no_result<PMS0(volatile S&)>();
    test_no_result<PMS0(const volatile S&)>();
    test_no_result<PMS0(ND&)>();
    test_no_result<PMS0(const ND&)>();

#ifdef _LIBCUDACXX_HAS_MEMORY
    test_no_result<PMS0(cuda::std::unique_ptr<S const>)>();
    test_no_result<PMS0(cuda::std::unique_ptr<ND>)>();
#endif // _LIBCUDACXX_HAS_MEMORY
    test_no_result<PMS0(cuda::std::reference_wrapper<S const>)>();
    test_no_result<PMS0(cuda::std::reference_wrapper<ND>)>();

    test_result_of<PMS1(S, int), int*>();
    test_result_of<PMS1(S&, int), int*>();
    test_result_of<PMS1(S*, int), int*>();
    test_result_of<PMS1(S*&, int), int*>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMS1(cuda::std::unique_ptr<S>, int), int*>();
    test_result_of<PMS1(cuda::std::unique_ptr<SD>, int), int*>();
#endif // _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMS1(cuda::std::reference_wrapper<S>, int), int*>();
    test_result_of<PMS1(const cuda::std::reference_wrapper<S>&, int), int*>();
    test_result_of<PMS1(cuda::std::reference_wrapper<SD>, int), int*>();
    test_result_of<PMS1(const cuda::std::reference_wrapper<SD>&, int), int*>();

    test_no_result<PMS1(const S&, int)>();
    test_no_result<PMS1(volatile S&, int)>();
    test_no_result<PMS1(const volatile S&, int)>();
    test_no_result<PMS1(ND&, int)>();
    test_no_result<PMS1(const ND&, int)>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    test_no_result<PMS1(cuda::std::unique_ptr<S const>, int)>();
    test_no_result<PMS1(cuda::std::unique_ptr<ND>, int)>();
#endif // _LIBCUDACXX_HAS_MEMORY
    test_no_result<PMS1(cuda::std::reference_wrapper<S const>, int)>();
    test_no_result<PMS1(cuda::std::reference_wrapper<ND>, int)>();

    test_result_of<PMS2(S, int, int), int&>();
    test_result_of<PMS2(S&, int, int), int&>();
    test_result_of<PMS2(S*, int, int), int&>();
    test_result_of<PMS2(S*&, int, int), int&>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMS2(cuda::std::unique_ptr<S>, int, int), int&>();
    test_result_of<PMS2(cuda::std::unique_ptr<SD>, int, int), int&>();
#endif // _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMS2(cuda::std::reference_wrapper<S>, int, int), int&>();
    test_result_of<PMS2(const cuda::std::reference_wrapper<S>&, int, int), int&>();
    test_result_of<PMS2(cuda::std::reference_wrapper<SD>, int, int), int&>();
    test_result_of<PMS2(const cuda::std::reference_wrapper<SD>&, int, int), int&>();

    test_no_result<PMS2(const S&, int, int)>();
    test_no_result<PMS2(volatile S&, int, int)>();
    test_no_result<PMS2(const volatile S&, int, int)>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    test_no_result<PMS2(cuda::std::unique_ptr<S const>, int, int)>();
#endif // _LIBCUDACXX_HAS_MEMORY
    test_no_result<PMS2(cuda::std::reference_wrapper<S const>, int, int)>();

    test_no_result<PMS2(const ND&, int, int)>();
    test_no_result<PMS2(cuda::std::reference_wrapper<ND>, int, int)>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    test_no_result<PMS2(cuda::std::unique_ptr<ND>, int, int)>();
#endif // _LIBCUDACXX_HAS_MEMORY

    test_result_of<PMS3(S&, int), const int&>();
    test_result_of<PMS3(S&, int, long), const int&>();

    typedef int (S::*PMS0C)() const;
    typedef int* (S::*PMS1C)(long) const;
    typedef int& (S::*PMS2C)(long, int) const;
    typedef const int& (S::*PMS3C)(int, ...) const;
    test_result_of<PMS0C(S), int>();
    test_result_of<PMS0C(S&), int>();
    test_result_of<PMS0C(const S&), int>();
    test_result_of<PMS0C(S*), int>();
    test_result_of<PMS0C(const S*), int>();
    test_result_of<PMS0C(S*&), int>();
    test_result_of<PMS0C(const S*&), int>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMS0C(cuda::std::unique_ptr<S>), int>();
    test_result_of<PMS0C(cuda::std::unique_ptr<SD>), int>();
#endif // _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMS0C(cuda::std::reference_wrapper<S>), int>();
    test_result_of<PMS0C(cuda::std::reference_wrapper<const S>), int>();
    test_result_of<PMS0C(const cuda::std::reference_wrapper<S>&), int>();
    test_result_of<PMS0C(const cuda::std::reference_wrapper<const S>&), int>();
    test_result_of<PMS0C(cuda::std::reference_wrapper<SD>), int>();
    test_result_of<PMS0C(cuda::std::reference_wrapper<const SD>), int>();
    test_result_of<PMS0C(const cuda::std::reference_wrapper<SD>&), int>();
    test_result_of<PMS0C(const cuda::std::reference_wrapper<const SD>&), int>();

    test_no_result<PMS0C(volatile S&)>();
    test_no_result<PMS0C(const volatile S&)>();

    test_result_of<PMS1C(S, int), int*>();
    test_result_of<PMS1C(S&, int), int*>();
    test_result_of<PMS1C(const S&, int), int*>();
    test_result_of<PMS1C(S*, int), int*>();
    test_result_of<PMS1C(const S*, int), int*>();
    test_result_of<PMS1C(S*&, int), int*>();
    test_result_of<PMS1C(const S*&, int), int*>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMS1C(cuda::std::unique_ptr<S>, int), int*>();
#endif // _LIBCUDACXX_HAS_MEMORY
    test_no_result<PMS1C(volatile S&, int)>();
    test_no_result<PMS1C(const volatile S&, int)>();

    test_result_of<PMS2C(S, int, int), int&>();
    test_result_of<PMS2C(S&, int, int), int&>();
    test_result_of<PMS2C(const S&, int, int), int&>();
    test_result_of<PMS2C(S*, int, int), int&>();
    test_result_of<PMS2C(const S*, int, int), int&>();
    test_result_of<PMS2C(S*&, int, int), int&>();
    test_result_of<PMS2C(const S*&, int, int), int&>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMS2C(cuda::std::unique_ptr<S>, int, int), int&>();
#endif // _LIBCUDACXX_HAS_MEMORY
    test_no_result<PMS2C(volatile S&, int, int)>();
    test_no_result<PMS2C(const volatile S&, int, int)>();

    test_result_of<PMS3C(S&, int), const int&>();
    test_result_of<PMS3C(S&, int, long), const int&>();

    typedef int (S::*PMS0V)() volatile;
    typedef int* (S::*PMS1V)(long) volatile;
    typedef int& (S::*PMS2V)(long, int) volatile;
    typedef const int& (S::*PMS3V)(int, ...) volatile;
    test_result_of<PMS0V(S), int>();
    test_result_of<PMS0V(S&), int>();
    test_result_of<PMS0V(volatile S&), int>();
    test_result_of<PMS0V(S*), int>();
    test_result_of<PMS0V(volatile S*), int>();
    test_result_of<PMS0V(S*&), int>();
    test_result_of<PMS0V(volatile S*&), int>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMS0V(cuda::std::unique_ptr<S>), int>();
#endif // _LIBCUDACXX_HAS_MEMORY
    test_no_result<PMS0V(const S&)>();
    test_no_result<PMS0V(const volatile S&)>();

    test_result_of<PMS1V(S, int), int*>();
    test_result_of<PMS1V(S&, int), int*>();
    test_result_of<PMS1V(volatile S&, int), int*>();
    test_result_of<PMS1V(S*, int), int*>();
    test_result_of<PMS1V(volatile S*, int), int*>();
    test_result_of<PMS1V(S*&, int), int*>();
    test_result_of<PMS1V(volatile S*&, int), int*>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMS1V(cuda::std::unique_ptr<S>, int), int*>();
#endif // _LIBCUDACXX_HAS_MEMORY
    test_no_result<PMS1V(const S&, int)>();
    test_no_result<PMS1V(const volatile S&, int)>();

    test_result_of<PMS2V(S, int, int), int&>();
    test_result_of<PMS2V(S&, int, int), int&>();
    test_result_of<PMS2V(volatile S&, int, int), int&>();
    test_result_of<PMS2V(S*, int, int), int&>();
    test_result_of<PMS2V(volatile S*, int, int), int&>();
    test_result_of<PMS2V(S*&, int, int), int&>();
    test_result_of<PMS2V(volatile S*&, int, int), int&>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMS2V(cuda::std::unique_ptr<S>, int, int), int&>();
#endif // _LIBCUDACXX_HAS_MEMORY
    test_no_result<PMS2V(const S&, int, int)>();
    test_no_result<PMS2V(const volatile S&, int, int)>();

    test_result_of<PMS3V(S&, int), const int&>();
    test_result_of<PMS3V(S&, int, long), const int&>();

    typedef int (S::*PMS0CV)() const volatile;
    typedef int* (S::*PMS1CV)(long) const volatile;
    typedef int& (S::*PMS2CV)(long, int) const volatile;
    typedef const int& (S::*PMS3CV)(int, ...) const volatile;
    test_result_of<PMS0CV(S), int>();
    test_result_of<PMS0CV(S&), int>();
    test_result_of<PMS0CV(const S&), int>();
    test_result_of<PMS0CV(volatile S&), int>();
    test_result_of<PMS0CV(const volatile S&), int>();
    test_result_of<PMS0CV(S*), int>();
    test_result_of<PMS0CV(const S*), int>();
    test_result_of<PMS0CV(volatile S*), int>();
    test_result_of<PMS0CV(const volatile S*), int>();
    test_result_of<PMS0CV(S*&), int>();
    test_result_of<PMS0CV(const S*&), int>();
    test_result_of<PMS0CV(volatile S*&), int>();
    test_result_of<PMS0CV(const volatile S*&), int>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMS0CV(cuda::std::unique_ptr<S>), int>();
#endif // _LIBCUDACXX_HAS_MEMORY

    test_result_of<PMS1CV(S, int), int*>();
    test_result_of<PMS1CV(S&, int), int*>();
    test_result_of<PMS1CV(const S&, int), int*>();
    test_result_of<PMS1CV(volatile S&, int), int*>();
    test_result_of<PMS1CV(const volatile S&, int), int*>();
    test_result_of<PMS1CV(S*, int), int*>();
    test_result_of<PMS1CV(const S*, int), int*>();
    test_result_of<PMS1CV(volatile S*, int), int*>();
    test_result_of<PMS1CV(const volatile S*, int), int*>();
    test_result_of<PMS1CV(S*&, int), int*>();
    test_result_of<PMS1CV(const S*&, int), int*>();
    test_result_of<PMS1CV(volatile S*&, int), int*>();
    test_result_of<PMS1CV(const volatile S*&, int), int*>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMS1CV(cuda::std::unique_ptr<S>, int), int*>();
#endif // _LIBCUDACXX_HAS_MEMORY

    test_result_of<PMS2CV(S, int, int), int&>();
    test_result_of<PMS2CV(S&, int, int), int&>();
    test_result_of<PMS2CV(const S&, int, int), int&>();
    test_result_of<PMS2CV(volatile S&, int, int), int&>();
    test_result_of<PMS2CV(const volatile S&, int, int), int&>();
    test_result_of<PMS2CV(S*, int, int), int&>();
    test_result_of<PMS2CV(const S*, int, int), int&>();
    test_result_of<PMS2CV(volatile S*, int, int), int&>();
    test_result_of<PMS2CV(const volatile S*, int, int), int&>();
    test_result_of<PMS2CV(S*&, int, int), int&>();
    test_result_of<PMS2CV(const S*&, int, int), int&>();
    test_result_of<PMS2CV(volatile S*&, int, int), int&>();
    test_result_of<PMS2CV(const volatile S*&, int, int), int&>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMS2CV(cuda::std::unique_ptr<S>, int, int), int&>();
#endif // _LIBCUDACXX_HAS_MEMORY

    test_result_of<PMS3CV(S&, int), const int&>();
    test_result_of<PMS3CV(S&, int, long), const int&>();
  }
  { // pointer to member data
    typedef char S::*PMD;
    test_result_of<PMD(S&), char&>();
    test_result_of<PMD(S*), char&>();
    test_result_of<PMD(S* const), char&>();
    test_result_of<PMD(const S&), const char&>();
    test_result_of<PMD(const S*), const char&>();
    test_result_of<PMD(volatile S&), volatile char&>();
    test_result_of<PMD(volatile S*), volatile char&>();
    test_result_of<PMD(const volatile S&), const volatile char&>();
    test_result_of<PMD(const volatile S*), const volatile char&>();
    test_result_of<PMD(SD&), char&>();
    test_result_of<PMD(SD const&), const char&>();
    test_result_of<PMD(SD*), char&>();
    test_result_of<PMD(const SD*), const char&>();
#ifdef _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMD(cuda::std::unique_ptr<S>), char&>();
    test_result_of<PMD(cuda::std::unique_ptr<S const>), const char&>();
#endif // _LIBCUDACXX_HAS_MEMORY
    test_result_of<PMD(cuda::std::reference_wrapper<S>), char&>();
    test_result_of<PMD(cuda::std::reference_wrapper<S const>), const char&>();
    test_no_result<PMD(ND&)>();
  }
#if defined(TEST_COMPILER_NVCC)
  { // extended lambda
    NV_IF_TARGET(
      NV_IS_DEVICE,
      (test_lambda<int>([] __host__ __device__() -> int {
         return 42;
       });
       test_lambda<double>([] __host__ __device__() -> double {
         return 42.0;
       });
       test_lambda<SD>([] __host__ __device__() -> SD {
         return {};
       });))
    test_lambda<double>(cuda::proclaim_return_type<double>([] __device__() -> double {
      return 42.0;
    }));
  }
#endif

  return 0;
}
