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
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_IGNORE_DEPRECATED_API

#include <cuda/std/type_traits>
#ifdef _LIBCUDACXX_HAS_MEMORY
#  include <cuda/std/memory>
#endif // _LIBCUDACXX_HAS_MEMORY
#include <cuda/functional>
#include <cuda/std/cassert>

#include "test_macros.h"

struct S
{
  using FreeFunc = short (*)(long);
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
  using type = void;
};

template <class T, class = void>
struct HasType : cuda::std::false_type
{};

template <class T>
struct HasType<T, typename Voider<typename T::type>::type> : cuda::std::true_type
{};

template <typename T, typename U>
struct test_invoke_result;

template <typename Fn, typename... Args, typename Ret>
struct test_invoke_result<Fn(Args...), Ret>
{
  __host__ __device__ static void call()
  {
    static_assert(cuda::std::is_invocable<Fn, Args...>::value, "");
    static_assert(cuda::std::is_invocable_r<Ret, Fn, Args...>::value, "");
    static_assert(cuda::std::is_same_v<Ret, typename cuda::std::invoke_result<Fn, Args...>::type>);
  }
};

template <class T, class U>
__host__ __device__ void test_result_of()
{
  static_assert(cuda::std::is_same_v<U, typename cuda::std::result_of<T>::type>);
  test_invoke_result<T, U>::call();
}

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

template <class T>
__host__ __device__ void test_no_result()
{
  static_assert((!HasType<cuda::std::result_of<T>>::value), "");
  test_invoke_no_result<T>::call();
}

#if TEST_CUDA_COMPILER(NVCC)
template <class Ret, class Fn>
__host__ __device__ void test_lambda(Fn&&)
{
  static_assert(cuda::std::is_same_v<Ret, typename cuda::std::result_of<Fn()>::type>);
  static_assert(cuda::std::is_same_v<Ret, typename cuda::std::invoke_result<Fn>::type>);
}
#endif // TEST_CUDA_COMPILER(NVCC)

int main(int, char**)
{
  using ND = NotDerived;
  { // functor object
    test_result_of<S(int), short>();
    test_result_of<S&(unsigned char, int&), double>();
    test_result_of<S const&(unsigned char, int&), double const&>();
    test_result_of<S volatile&(unsigned char, int&), double volatile&>();
    test_result_of<S const volatile&(unsigned char, int&), double const volatile&>();
  }
  { // pointer to function
    using RF0  = bool (&)();
    using RF1  = bool* (&) (int);
    using RF2  = bool& (&) (int, int);
    using RF3  = bool const& (&) (int, int, int);
    using RF4  = bool (&)(int, ...);
    using PF0  = bool (*)();
    using PF1  = bool* (*) (int);
    using PF2  = bool& (*) (int, int);
    using PF3  = bool const& (*) (int, int, int);
    using PF4  = bool (*)(int, ...);
    using PRF0 = bool (*&)();
    using PRF1 = bool* (*&) (int);
    using PRF2 = bool& (*&) (int, int);
    using PRF3 = bool const& (*&) (int, int, int);
    using PRF4 = bool (*&)(int, ...);
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

    using PMS0 = int (S::*)();
    using PMS1 = int* (S::*) (long);
    using PMS2 = int& (S::*) (long, int);
    using PMS3 = const int& (S::*) (int, ...);
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

    using PMS0C = int (S::*)() const;
    using PMS1C = int* (S::*) (long) const;
    using PMS2C = int& (S::*) (long, int) const;
    using PMS3C = const int& (S::*) (int, ...) const;
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

    using PMS0V = int (S::*)() volatile;
    using PMS1V = int* (S::*) (long) volatile;
    using PMS2V = int& (S::*) (long, int) volatile;
    using PMS3V = const int& (S::*) (int, ...) volatile;
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

    using PMS0CV = int (S::*)() const volatile;
    using PMS1CV = int* (S::*) (long) const volatile;
    using PMS2CV = int& (S::*) (long, int) const volatile;
    using PMS3CV = const int& (S::*) (int, ...) const volatile;
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
    using PMD = char S::*;
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
#if TEST_CUDA_COMPILER(NVCC)
  { // extended lambda
#  if _CCCL_CUDACC_AT_LEAST(12, 3)
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
#  endif // _CCCL_CUDACC_AT_LEAST(12, 3)
    test_lambda<double>(cuda::proclaim_return_type<double>([] __device__() -> double {
      return 42.0;
    }));
  }
#endif // TEST_CUDA_COMPILER(NVCC)

  return 0;
}
