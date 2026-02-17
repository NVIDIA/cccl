//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

_CCCL_DIAG_SUPPRESS_MSVC(4594) // class C can never be instantiated - indirect virtual base class D is inaccessible
_CCCL_DIAG_SUPPRESS_MSVC(4624) // class destructor was implicitly defined as deleted

template <class Base, class Derived>
__host__ __device__ constexpr void test_is_virtual_base_of(bool expected)
{
#if defined(_CCCL_BUILTIN_IS_VIRTUAL_BASE_OF)
  // Test the type of the variables
  {
    static_assert(cuda::std::is_same_v<bool const, decltype(cuda::std::is_virtual_base_of<Base, Derived>::value)>);
    static_assert(cuda::std::is_same_v<bool const, decltype(cuda::std::is_virtual_base_of_v<Base, Derived>)>);
  }

  // Test their value
  {
    assert((cuda::std::is_virtual_base_of<Base, Derived>::value == expected));
    assert((cuda::std::is_virtual_base_of<Base, const Derived>::value == expected));
    assert((cuda::std::is_virtual_base_of<Base, volatile Derived>::value == expected));
    assert((cuda::std::is_virtual_base_of<Base, const volatile Derived>::value == expected));
    assert((cuda::std::is_virtual_base_of<const Base, Derived>::value == expected));
    assert((cuda::std::is_virtual_base_of<const Base, const Derived>::value == expected));
    assert((cuda::std::is_virtual_base_of<const Base, volatile Derived>::value == expected));
    assert((cuda::std::is_virtual_base_of<const Base, const volatile Derived>::value == expected));
    assert((cuda::std::is_virtual_base_of<volatile Base, Derived>::value == expected));
    assert((cuda::std::is_virtual_base_of<volatile Base, const Derived>::value == expected));
    assert((cuda::std::is_virtual_base_of<volatile Base, volatile Derived>::value == expected));
    assert((cuda::std::is_virtual_base_of<volatile Base, const volatile Derived>::value == expected));
    assert((cuda::std::is_virtual_base_of<const volatile Base, Derived>::value == expected));
    assert((cuda::std::is_virtual_base_of<const volatile Base, const Derived>::value == expected));
    assert((cuda::std::is_virtual_base_of<const volatile Base, volatile Derived>::value == expected));
    assert((cuda::std::is_virtual_base_of<const volatile Base, const volatile Derived>::value == expected));

    assert((cuda::std::is_virtual_base_of_v<Base, Derived> == expected));
    assert((cuda::std::is_virtual_base_of_v<Base, const Derived> == expected));
    assert((cuda::std::is_virtual_base_of_v<Base, volatile Derived> == expected));
    assert((cuda::std::is_virtual_base_of_v<Base, const volatile Derived> == expected));
    assert((cuda::std::is_virtual_base_of_v<const Base, Derived> == expected));
    assert((cuda::std::is_virtual_base_of_v<const Base, const Derived> == expected));
    assert((cuda::std::is_virtual_base_of_v<const Base, volatile Derived> == expected));
    assert((cuda::std::is_virtual_base_of_v<const Base, const volatile Derived> == expected));
    assert((cuda::std::is_virtual_base_of_v<volatile Base, Derived> == expected));
    assert((cuda::std::is_virtual_base_of_v<volatile Base, const Derived> == expected));
    assert((cuda::std::is_virtual_base_of_v<volatile Base, volatile Derived> == expected));
    assert((cuda::std::is_virtual_base_of_v<volatile Base, const volatile Derived> == expected));
    assert((cuda::std::is_virtual_base_of_v<const volatile Base, Derived> == expected));
    assert((cuda::std::is_virtual_base_of_v<const volatile Base, const Derived> == expected));
    assert((cuda::std::is_virtual_base_of_v<const volatile Base, volatile Derived> == expected));
    assert((cuda::std::is_virtual_base_of_v<const volatile Base, const volatile Derived> == expected));
  }

  // Check the relationship with is_base_of. If it's not a base of, it can't be a virtual base of.
  {
    assert((!cuda::std::is_base_of_v<Base, Derived> ? !cuda::std::is_virtual_base_of_v<Base, Derived> : true));
  }

  // Make sure they can be referenced at runtime
  {
    bool const& a = cuda::std::is_virtual_base_of<Base, Derived>::value;
    bool const& b = cuda::std::is_virtual_base_of_v<Base, Derived>;
    assert(a == expected);
    assert(b == expected);
  }
#endif // defined(_CCCL_BUILTIN_IS_VIRTUAL_BASE_OF)
}

struct Incomplete;
struct Unrelated
{};
union IncompleteUnion;
union Union
{
  int i;
  float f;
};

class Base
{};
class Derived : Base
{};
class Derived2 : Base
{};
class Derived2a : Derived
{};
class Derived2b : Derived
{};
class Derived3Virtual
    : virtual Derived2a
    , virtual Derived2b
{};

struct DerivedTransitiveViaNonVirtual : Derived3Virtual
{};
struct DerivedTransitiveViaVirtual : virtual Derived3Virtual
{};

template <typename T>
struct CrazyDerived : T
{};
template <typename T>
struct CrazyDerivedVirtual : virtual T
{};

struct DerivedPrivate : private virtual Base
{};
struct DerivedProtected : protected virtual Base
{};
struct DerivedPrivatePrivate : private DerivedPrivate
{};
struct DerivedPrivateProtected : private DerivedProtected
{};
struct DerivedProtectedPrivate : protected DerivedProtected
{};
struct DerivedProtectedProtected : protected DerivedProtected
{};
struct DerivedTransitivePrivate
    : private Derived
    , private Derived2
{};

__host__ __device__ constexpr bool test()
{
  // Test with non-virtual inheritance
  {
    test_is_virtual_base_of<Base, Base>(false);
    test_is_virtual_base_of<Base, Derived>(false);
    test_is_virtual_base_of<Base, Derived2>(false);
    test_is_virtual_base_of<Derived, DerivedTransitivePrivate>(false);
    test_is_virtual_base_of<Derived, Base>(false);
    test_is_virtual_base_of<Incomplete, Derived>(false);
  }

  // Test with virtual inheritance
  {
    test_is_virtual_base_of<Base, Derived3Virtual>(false);
    test_is_virtual_base_of<Derived, Derived3Virtual>(false);
    test_is_virtual_base_of<Derived2b, Derived3Virtual>(true);
    test_is_virtual_base_of<Derived2a, Derived3Virtual>(true);
    test_is_virtual_base_of<Base, DerivedPrivate>(true);
    test_is_virtual_base_of<Base, DerivedProtected>(true);
    test_is_virtual_base_of<Base, DerivedPrivatePrivate>(true);
    test_is_virtual_base_of<Base, DerivedPrivateProtected>(true);
    test_is_virtual_base_of<Base, DerivedProtectedPrivate>(true);
    test_is_virtual_base_of<Base, DerivedProtectedProtected>(true);
    test_is_virtual_base_of<Derived2a, DerivedTransitiveViaNonVirtual>(true);
    test_is_virtual_base_of<Derived2b, DerivedTransitiveViaNonVirtual>(true);
    test_is_virtual_base_of<Derived2a, DerivedTransitiveViaVirtual>(true);
    test_is_virtual_base_of<Derived2b, DerivedTransitiveViaVirtual>(true);
    test_is_virtual_base_of<Base, CrazyDerived<Base>>(false);
    test_is_virtual_base_of<CrazyDerived<Base>, Base>(false);
    test_is_virtual_base_of<Base, CrazyDerivedVirtual<Base>>(true);
    test_is_virtual_base_of<CrazyDerivedVirtual<Base>, Base>(false);
  }

  // Test unrelated types
  {
    test_is_virtual_base_of<Base&, Derived&>(false);
    test_is_virtual_base_of<Base[3], Derived[3]>(false);
    test_is_virtual_base_of<Unrelated, Derived>(false);
    test_is_virtual_base_of<Base, Unrelated>(false);
    test_is_virtual_base_of<Base, void>(false);
    test_is_virtual_base_of<void, Derived>(false);
  }

  // Test scalar types
  {
    test_is_virtual_base_of<int, Base>(false);
    test_is_virtual_base_of<int, Derived>(false);
    test_is_virtual_base_of<int, Incomplete>(false);
    test_is_virtual_base_of<int, int>(false);

    test_is_virtual_base_of<Base, int>(false);
    test_is_virtual_base_of<Derived, int>(false);
    test_is_virtual_base_of<Incomplete, int>(false);

    test_is_virtual_base_of<int[], int[]>(false);
    test_is_virtual_base_of<long, int>(false);
    test_is_virtual_base_of<int, long>(false);
  }

  // Test unions
  {
    test_is_virtual_base_of<Union, Union>(false);
    test_is_virtual_base_of<IncompleteUnion, IncompleteUnion>(false);
    test_is_virtual_base_of<Union, IncompleteUnion>(false);
    test_is_virtual_base_of<IncompleteUnion, Union>(false);
    test_is_virtual_base_of<Incomplete, IncompleteUnion>(false);
    test_is_virtual_base_of<IncompleteUnion, Incomplete>(false);
    test_is_virtual_base_of<Unrelated, IncompleteUnion>(false);
    test_is_virtual_base_of<IncompleteUnion, Unrelated>(false);
    test_is_virtual_base_of<int, IncompleteUnion>(false);
    test_is_virtual_base_of<IncompleteUnion, int>(false);
    test_is_virtual_base_of<Unrelated, Union>(false);
    test_is_virtual_base_of<Union, Unrelated>(false);
    test_is_virtual_base_of<int, Unrelated>(false);
    test_is_virtual_base_of<Union, int>(false);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
