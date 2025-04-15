//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// template <class F, class ...Args>
// result_of_t<F&&(Args&&...)> invoke(F&&, Args&&...);

/// C++14 [func.def] 20.9.0
/// (1) The following definitions apply to this Clause:
/// (2) A call signature is the name of a return type followed by a parenthesized
///     comma-separated list of zero or more argument types.
/// (3) A callable type is a function object type (20.9) or a pointer to member.
/// (4) A callable object is an object of a callable type.
/// (5) A call wrapper type is a type that holds a callable object and supports
///     a call operation that forwards to that object.
/// (6) A call wrapper is an object of a call wrapper type.
/// (7) A target object is the callable object held by a call wrapper.

/// C++14 [func.require] 20.9.1
///
/// Define INVOKE (f, t1, t2, ..., tN) as follows:
///   (1.1) - (t1.*f)(t2, ..., tN) when f is a pointer to a member function of a class T and t1 is an object of
///   type T or a reference to an object of type T or a reference to an object of a type derived from T;
///   (1.2) - ((*t1).*f)(t2, ..., tN) when f is a pointer to a member function of a class T and t1 is not one of
///   the types described in the previous item;
///   (1.3) - t1.*f when N == 1 and f is a pointer to member data of a class T and t1 is an object of type T or a
///   reference to an object of type T or a reference to an object of a type derived from T;
///   (1.4) - (*t1).*f when N == 1 and f is a pointer to member data of a class T and t1 is not one of the types
///   described in the previous item;
///   (1.5) - f(t1, t2, ..., tN) in all other cases.

#define _LIBCUDACXX_ENABLE_CXX20_REMOVED_TYPE_TRAITS
// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/utility> // for cuda::std::move

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(set_but_not_used)

struct NonCopyable
{
  __host__ __device__ NonCopyable() {}

private:
  NonCopyable(NonCopyable const&)            = delete;
  NonCopyable& operator=(NonCopyable const&) = delete;
};

struct TestClass
{
  __host__ __device__ explicit TestClass(int x)
      : data(x)
  {}

  __host__ __device__ int& operator()(NonCopyable&&) &
  {
    return data;
  }
  __host__ __device__ int const& operator()(NonCopyable&&) const&
  {
    return data;
  }
  __host__ __device__ int volatile& operator()(NonCopyable&&) volatile&
  {
    return data;
  }
  __host__ __device__ int const volatile& operator()(NonCopyable&&) const volatile&
  {
    return data;
  }

  __host__ __device__ int&& operator()(NonCopyable&&) &&
  {
    return cuda::std::move(data);
  }
  __host__ __device__ int const&& operator()(NonCopyable&&) const&&
  {
    return cuda::std::move(data);
  }
  __host__ __device__ int volatile&& operator()(NonCopyable&&) volatile&&
  {
    return cuda::std::move(data);
  }
  __host__ __device__ int const volatile&& operator()(NonCopyable&&) const volatile&&
  {
    return cuda::std::move(data);
  }

  int data;

private:
  TestClass(TestClass const&)            = delete;
  TestClass& operator=(TestClass const&) = delete;
};

struct DerivedFromTestClass : public TestClass
{
  __host__ __device__ explicit DerivedFromTestClass(int x)
      : TestClass(x)
  {}
};

__host__ __device__ int& foo(NonCopyable&&)
{
  static int data = 42;
  return data;
}

template <class Signature, class Expect, class Functor>
__host__ __device__ void test_b12(Functor&& f)
{
  // Create the callable object.
  typedef Signature TestClass::* ClassFunc;
  ClassFunc func_ptr = &TestClass::operator();

  // Create the dummy arg.
  NonCopyable arg;

  // Check that the deduced return type of invoke is what is expected.
  typedef decltype(cuda::std::invoke(func_ptr, cuda::std::forward<Functor>(f), cuda::std::move(arg))) DeducedReturnType;
  static_assert((cuda::std::is_same<DeducedReturnType, Expect>::value), "");

  // Check that result_of_t matches Expect.
  typedef typename cuda::std::result_of<ClassFunc && (Functor&&, NonCopyable&&)>::type ResultOfReturnType;
  static_assert((cuda::std::is_same<ResultOfReturnType, Expect>::value), "");

  // Run invoke and check the return value.
  DeducedReturnType ret = cuda::std::invoke(func_ptr, cuda::std::forward<Functor>(f), cuda::std::move(arg));
  assert(ret == 42);
}

template <class Expect, class Functor>
__host__ __device__ void test_b34(Functor&& f)
{
  // Create the callable object.
  typedef int TestClass::* ClassFunc;
  ClassFunc func_ptr = &TestClass::data;

  // Check that the deduced return type of invoke is what is expected.
  typedef decltype(cuda::std::invoke(func_ptr, cuda::std::forward<Functor>(f))) DeducedReturnType;
  static_assert((cuda::std::is_same<DeducedReturnType, Expect>::value), "");

  // Check that result_of_t matches Expect.
  typedef typename cuda::std::result_of<ClassFunc && (Functor&&)>::type ResultOfReturnType;
  static_assert((cuda::std::is_same<ResultOfReturnType, Expect>::value), "");

  // Run invoke and check the return value.
  DeducedReturnType ret = cuda::std::invoke(func_ptr, cuda::std::forward<Functor>(f));
  assert(ret == 42);
}

template <class Expect, class Functor>
__host__ __device__ void test_b5(Functor&& f)
{
  NonCopyable arg;

  // Check that the deduced return type of invoke is what is expected.
  typedef decltype(cuda::std::invoke(cuda::std::forward<Functor>(f), cuda::std::move(arg))) DeducedReturnType;
  static_assert((cuda::std::is_same<DeducedReturnType, Expect>::value), "");

  // Check that result_of_t matches Expect.
  typedef typename cuda::std::result_of<Functor && (NonCopyable&&)>::type ResultOfReturnType;
  static_assert((cuda::std::is_same<ResultOfReturnType, Expect>::value), "");

  // Run invoke and check the return value.
  DeducedReturnType ret = cuda::std::invoke(cuda::std::forward<Functor>(f), cuda::std::move(arg));
  assert(ret == 42);
}

__host__ __device__ void bullet_one_two_tests()
{
  {
    TestClass cl(42);
    test_b12<int&(NonCopyable&&) &, int&>(cl);
    test_b12<int const&(NonCopyable&&) const&, int const&>(cl);
    test_b12<int volatile&(NonCopyable&&) volatile&, int volatile&>(cl);
    test_b12<int const volatile&(NonCopyable&&) const volatile&, int const volatile&>(cl);

    test_b12<int && (NonCopyable&&)&&, int&&>(cuda::std::move(cl));
    test_b12<int const && (NonCopyable&&) const&&, int const&&>(cuda::std::move(cl));
    test_b12<int volatile && (NonCopyable&&) volatile&&, int volatile&&>(cuda::std::move(cl));
    test_b12<int const volatile && (NonCopyable&&) const volatile&&, int const volatile&&>(cuda::std::move(cl));
  }
  {
    DerivedFromTestClass cl(42);
    test_b12<int&(NonCopyable&&) &, int&>(cl);
    test_b12<int const&(NonCopyable&&) const&, int const&>(cl);
    test_b12<int volatile&(NonCopyable&&) volatile&, int volatile&>(cl);
    test_b12<int const volatile&(NonCopyable&&) const volatile&, int const volatile&>(cl);

    test_b12<int && (NonCopyable&&)&&, int&&>(cuda::std::move(cl));
    test_b12<int const && (NonCopyable&&) const&&, int const&&>(cuda::std::move(cl));
    test_b12<int volatile && (NonCopyable&&) volatile&&, int volatile&&>(cuda::std::move(cl));
    test_b12<int const volatile && (NonCopyable&&) const volatile&&, int const volatile&&>(cuda::std::move(cl));
  }
#ifndef __cuda_std__
  // uncomment when reenabling reference_wrapper
  {
    TestClass cl_obj(42);
    cuda::std::reference_wrapper<TestClass> cl(cl_obj);
    test_b12<int&(NonCopyable&&) &, int&>(cl);
    test_b12<int const&(NonCopyable&&) const&, int const&>(cl);
    test_b12<int volatile&(NonCopyable&&) volatile&, int volatile&>(cl);
    test_b12<int const volatile&(NonCopyable&&) const volatile&, int const volatile&>(cl);

    test_b12<int&(NonCopyable&&) &, int&>(cuda::std::move(cl));
    test_b12<int const&(NonCopyable&&) const&, int const&>(cuda::std::move(cl));
    test_b12<int volatile&(NonCopyable&&) volatile&, int volatile&>(cuda::std::move(cl));
    test_b12<int const volatile&(NonCopyable&&) const volatile&, int const volatile&>(cuda::std::move(cl));
  }
  {
    DerivedFromTestClass cl_obj(42);
    cuda::std::reference_wrapper<DerivedFromTestClass> cl(cl_obj);
    test_b12<int&(NonCopyable&&) &, int&>(cl);
    test_b12<int const&(NonCopyable&&) const&, int const&>(cl);
    test_b12<int volatile&(NonCopyable&&) volatile&, int volatile&>(cl);
    test_b12<int const volatile&(NonCopyable&&) const volatile&, int const volatile&>(cl);

    test_b12<int&(NonCopyable&&) &, int&>(cuda::std::move(cl));
    test_b12<int const&(NonCopyable&&) const&, int const&>(cuda::std::move(cl));
    test_b12<int volatile&(NonCopyable&&) volatile&, int volatile&>(cuda::std::move(cl));
    test_b12<int const volatile&(NonCopyable&&) const volatile&, int const volatile&>(cuda::std::move(cl));
  }
#endif
  {
    TestClass cl_obj(42);
    TestClass* cl = &cl_obj;
    test_b12<int&(NonCopyable&&) &, int&>(cl);
    test_b12<int const&(NonCopyable&&) const&, int const&>(cl);
    test_b12<int volatile&(NonCopyable&&) volatile&, int volatile&>(cl);
    test_b12<int const volatile&(NonCopyable&&) const volatile&, int const volatile&>(cl);
  }
  {
    DerivedFromTestClass cl_obj(42);
    DerivedFromTestClass* cl = &cl_obj;
    test_b12<int&(NonCopyable&&) &, int&>(cl);
    test_b12<int const&(NonCopyable&&) const&, int const&>(cl);
    test_b12<int volatile&(NonCopyable&&) volatile&, int volatile&>(cl);
    test_b12<int const volatile&(NonCopyable&&) const volatile&, int const volatile&>(cl);
  }
}

__host__ __device__ void bullet_three_four_tests()
{
  {
    typedef TestClass Fn;
    Fn cl(42);
    test_b34<int&>(cl);
    test_b34<int const&>(static_cast<Fn const&>(cl));
    test_b34<int volatile&>(static_cast<Fn volatile&>(cl));
    test_b34<int const volatile&>(static_cast<Fn const volatile&>(cl));

    test_b34<int&&>(static_cast<Fn&&>(cl));
    test_b34<int const&&>(static_cast<Fn const&&>(cl));
    test_b34<int volatile&&>(static_cast<Fn volatile&&>(cl));
    test_b34<int const volatile&&>(static_cast<Fn const volatile&&>(cl));
  }
  {
    typedef DerivedFromTestClass Fn;
    Fn cl(42);
    test_b34<int&>(cl);
    test_b34<int const&>(static_cast<Fn const&>(cl));
    test_b34<int volatile&>(static_cast<Fn volatile&>(cl));
    test_b34<int const volatile&>(static_cast<Fn const volatile&>(cl));

    test_b34<int&&>(static_cast<Fn&&>(cl));
    test_b34<int const&&>(static_cast<Fn const&&>(cl));
    test_b34<int volatile&&>(static_cast<Fn volatile&&>(cl));
    test_b34<int const volatile&&>(static_cast<Fn const volatile&&>(cl));
  }
#ifndef __cuda_std__
  // uncomment when reenabling reference_wrapper
  {
    typedef TestClass Fn;
    Fn cl(42);
    test_b34<int&>(cuda::std::reference_wrapper<Fn>(cl));
    test_b34<int const&>(cuda::std::reference_wrapper<Fn const>(cl));
    test_b34<int volatile&>(cuda::std::reference_wrapper<Fn volatile>(cl));
    test_b34<int const volatile&>(cuda::std::reference_wrapper<Fn const volatile>(cl));
  }
  {
    typedef DerivedFromTestClass Fn;
    Fn cl(42);
    test_b34<int&>(cuda::std::reference_wrapper<Fn>(cl));
    test_b34<int const&>(cuda::std::reference_wrapper<Fn const>(cl));
    test_b34<int volatile&>(cuda::std::reference_wrapper<Fn volatile>(cl));
    test_b34<int const volatile&>(cuda::std::reference_wrapper<Fn const volatile>(cl));
  }
#endif
  {
    typedef TestClass Fn;
    Fn cl_obj(42);
    Fn* cl = &cl_obj;
    test_b34<int&>(cl);
    test_b34<int const&>(static_cast<Fn const*>(cl));
    test_b34<int volatile&>(static_cast<Fn volatile*>(cl));
    test_b34<int const volatile&>(static_cast<Fn const volatile*>(cl));
  }
  {
    typedef DerivedFromTestClass Fn;
    Fn cl_obj(42);
    Fn* cl = &cl_obj;
    test_b34<int&>(cl);
    test_b34<int const&>(static_cast<Fn const*>(cl));
    test_b34<int volatile&>(static_cast<Fn volatile*>(cl));
    test_b34<int const volatile&>(static_cast<Fn const volatile*>(cl));
  }
}

__host__ __device__ void bullet_five_tests()
{
  using FooType = int&(NonCopyable&&);
  {
    FooType& fn = foo;
    test_b5<int&>(fn);
  }
  {
    FooType* fn = foo;
    test_b5<int&>(fn);
  }
  {
    typedef TestClass Fn;
    Fn cl(42);
    test_b5<int&>(cl);
    test_b5<int const&>(static_cast<Fn const&>(cl));
    test_b5<int volatile&>(static_cast<Fn volatile&>(cl));
    test_b5<int const volatile&>(static_cast<Fn const volatile&>(cl));

    test_b5<int&&>(static_cast<Fn&&>(cl));
    test_b5<int const&&>(static_cast<Fn const&&>(cl));
    test_b5<int volatile&&>(static_cast<Fn volatile&&>(cl));
    test_b5<int const volatile&&>(static_cast<Fn const volatile&&>(cl));
  }
}

struct CopyThrows
{
  __host__ __device__ CopyThrows() {}
  __host__ __device__ CopyThrows(CopyThrows const&) {}
  __host__ __device__ CopyThrows(CopyThrows&&) noexcept {}
};

struct NoThrowCallable
{
  __host__ __device__ void operator()() noexcept {}
  __host__ __device__ void operator()(CopyThrows) noexcept {}
};

struct ThrowsCallable
{
  __host__ __device__ void operator()() {}
};

struct MemberObj
{
  int x;
};

__host__ __device__ void noexcept_test()
{
  {
    NoThrowCallable obj;
    unused(obj); // suppress unused warning
    CopyThrows arg;
    unused(arg); // suppress unused warning
    static_assert(noexcept(cuda::std::invoke(obj)), "");
#if !TEST_COMPILER(NVHPC)
    static_assert(!noexcept(cuda::std::invoke(obj, arg)), "");
#endif // TEST_COMPILER(NVHPC)
    static_assert(noexcept(cuda::std::invoke(obj, cuda::std::move(arg))), "");
  }
#if !TEST_COMPILER(NVHPC)
  {
    ThrowsCallable obj;
    unused(obj); // suppress unused warning
    static_assert(!noexcept(cuda::std::invoke(obj)), "");
  }
#endif // TEST_COMPILER(NVHPC)
  {
    MemberObj obj{42};
    unused(obj); // suppress unused warning.
    static_assert(noexcept(cuda::std::invoke(&MemberObj::x, obj)), "");
  }
}

int main(int, char**)
{
  bullet_one_two_tests();
  bullet_three_four_tests();
  bullet_five_tests();
  noexcept_test();

  return 0;
}
