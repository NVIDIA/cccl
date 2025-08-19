//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__utility/basic_any.h>
#include <cuda/__utility/immovable.h>
#include <cuda/std/__utility/as_const.h>
#include <cuda/std/__utility/typeid.h>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h" // IWYU pragma: keep

using immovable = cuda::__immovable;

struct TestCounters
{
  int objects = 0;
};

template <class TestType>
struct BasicAnyTestsFixture : TestCounters
{
  BasicAnyTestsFixture() = default;
};

template <class...>
struct iempty : cuda::__basic_interface<iempty>
{};

static_assert(cuda::__extension_of<iempty<>, iempty<>>);
static_assert(cuda::__extension_of<iempty<>, cuda::__iunknown>);

template <class...>
struct ibase : cuda::__basic_interface<ibase, cuda::__extends<cuda::__imovable<>>>
{
  _CCCL_HOST_DEVICE int foo(int i)
  {
    return cuda::__virtcall<&ibase::foo>(this, i);
  }

  template <class T>
  using overrides = cuda::__overrides_for<T, &T::foo>;
};

template <class...>
struct iderived : cuda::__basic_interface<iderived, cuda::__extends<ibase<>, cuda::__icopyable<>>>
{
  _CCCL_HOST_DEVICE int bar(int i)
  {
    return cuda::__virtcall<&iderived::bar>(this, i);
  }

  template <class T>
  using overrides = cuda::__overrides_for<T, &T::bar>;
};

template <bool Small>
struct SmallOrLarge
{
  unsigned int cookie = 0xDEADBEEF;
};

template <>
struct SmallOrLarge<false>
{
  unsigned int cookie = 0xDEADBEEF;
  ::cuda::std::byte buffer[cuda::__default_small_object_size]{};
};

constexpr bool Small = true;
constexpr bool Large = false;

using SmallType = SmallOrLarge<Small>;
using LargeType = SmallOrLarge<Large>;

template <bool Small>
struct Foo
{
  _CCCL_HOST_DEVICE Foo(int i, TestCounters* c)
      : j(i)
      , counters(c)
  {
    ++counters->objects;
  }

  _CCCL_HOST_DEVICE Foo(Foo&& other) noexcept
      : j(other.j)
      , counters(other.counters)
  {
    ++counters->objects;
    other.j = INT_MAX;
  }

  _CCCL_HOST_DEVICE Foo(Foo const& other) noexcept // TODO: test that types with throwing moves are "large"
      : j(other.j)
      , counters(other.counters)
  {
    ++counters->objects;
  }

  _CCCL_HOST_DEVICE ~Foo()
  {
    --counters->objects;
  }

  _CCCL_HOST_DEVICE Foo& operator=(Foo&& other) noexcept
  {
    operator=(other);
    other.j = INT_MAX;
    return *this;
  }

  _CCCL_HOST_DEVICE Foo& operator=(Foo const& other) noexcept
  {
    j        = other.j;
    counters = other.counters;
    return *this;
  }

  _CCCL_HOST_DEVICE int foo(int i)
  {
    return i + j;
  }

  int j;
  TestCounters* counters;
  SmallOrLarge<Small> k;
};

template <bool Small>
struct Bar : Foo<Small>
{
  using Foo<Small>::Foo;

  _CCCL_HOST_DEVICE int bar(int i)
  {
    return i * this->j;
  }
};

template <class...>
struct iregular
    : cuda::__basic_interface<iregular, cuda::__extends<cuda::__icopyable<>, cuda::__iequality_comparable<>>>
{};

struct Regular
{
  _CCCL_HOST_DEVICE bool operator==(Regular const& other) const
  {
    return i == other.i;
  }

  _CCCL_HOST_DEVICE bool operator!=(Regular const& other) const
  {
    return !operator==(other);
  }

  int i = 0;
};

struct any_regular : cuda::__basic_any<iregular<>>
{
  using cuda::__basic_any<iregular<>>::__basic_any;
};

struct any_regular_ref : cuda::__basic_any<iregular<>&>
{
  using cuda::__basic_any<iregular<>&>::__basic_any;
};

template <class TestType>
struct BasicAnyTest : BasicAnyTestsFixture<TestType>
{
  static constexpr bool IsSmall = ::cuda::std::is_same_v<TestType, SmallType>;

  _CCCL_HOST_DEVICE void test_type_traits()
  {
    static_assert(::cuda::std::is_standard_layout_v<cuda::__basic_any<iregular<>>>);
    static_assert(::cuda::std::is_standard_layout_v<cuda::__basic_any<iregular<>*>>);
    static_assert(::cuda::std::is_standard_layout_v<cuda::__basic_any<iregular<>&>>);
  }

  _CCCL_HOST_DEVICE void test_empty_interface_can_hold_anything()
  {
    static_assert(!::cuda::std::move_constructible<cuda::__basic_any<iempty<>>>);
    cuda::__basic_any<iempty<>> a{42};
    assert(a.has_value() == true);
    assert(a.type() == _CCCL_TYPEID(int));
    assert(a.interface() == _CCCL_TYPEID(iempty<>));
    assert(cuda::__any_cast<int>(&a));
    assert(cuda::__any_cast<int>(&a) == cuda::__any_cast<void>(&a));
    assert(*cuda::__any_cast<int>(&a) == 42);

    a.emplace<immovable>();
    assert(a.has_value() == true);
    assert(a.type() == _CCCL_TYPEID(immovable));
    assert(a.interface() == _CCCL_TYPEID(iempty<>));

    a.reset();
    assert(a.has_value() == false);
    assert(a.type() == _CCCL_TYPEID(void));
    assert(a.interface() == _CCCL_TYPEID(iempty<>));
  }

  _CCCL_HOST_DEVICE void test_interface_with_one_member_function()
  {
    static_assert(::cuda::std::move_constructible<cuda::__basic_any<ibase<>>>);
    static_assert(!::cuda::std::copy_constructible<cuda::__basic_any<ibase<>>>);

    cuda::__basic_any<ibase<>> a{::cuda::std::in_place_type<Foo<IsSmall>>, 42, this};
    assert(a.has_value() == true);
    assert(a.type() == _CCCL_TYPEID(Foo<IsSmall>));
    assert(a.interface() == _CCCL_TYPEID(ibase<>));
    assert(a.__in_situ() == IsSmall);
    assert(this->objects == 1);
    assert(a.foo(1) == 43);

    a.emplace<Foo<IsSmall>>(43, this);
    assert(a.foo(2) == 45);
    assert(this->objects == 1);

    a.reset();
    assert(a.has_value() == false);
    assert(a.type() == _CCCL_TYPEID(void));
    assert(a.interface() == _CCCL_TYPEID(ibase<>));
    assert(this->objects == 0);
  }

  _CCCL_HOST_DEVICE void test_single_interface_extension()
  {
    static_assert(::cuda::std::move_constructible<cuda::__basic_any<iderived<>>>);
    static_assert(::cuda::std::copy_constructible<cuda::__basic_any<iderived<>>>);

    cuda::__basic_any<iderived<>> a{::cuda::std::in_place_type<Bar<IsSmall>>, 42, this};
    assert(a.has_value() == true);
    assert(a.type() == _CCCL_TYPEID(Bar<IsSmall>));
    assert(a.interface() == _CCCL_TYPEID(iderived<>));
    assert(a.__in_situ() == IsSmall);
    assert(this->objects == 1);
    assert(a.foo(2) == 44);
    assert(a.bar(2) == 84);
    assert(cuda::__any_cast<Bar<IsSmall>>(&a)->j == 42);

    // construct a base any from a derived any:
    cuda::__basic_any<ibase<>> b{::cuda::std::move(a)};
    assert(b.has_value() == true);
    assert(b.type() == _CCCL_TYPEID(Bar<IsSmall>));
    assert(b.interface() == _CCCL_TYPEID(iderived<>));
    assert(b.__in_situ() == IsSmall);
    assert(this->objects == 1);
    assert(b.foo(1) == 43);
    assert(cuda::__any_cast<Bar<IsSmall>>(&b)->j == 42);

    a.emplace<Bar<IsSmall>>(-1, this);
    assert(a.foo(2) == 1);
    b = ::cuda::std::move(a);
    assert(b.foo(2) == 1);
    assert(cuda::__any_cast<Bar<IsSmall>>(&b)->j == -1);

    a.reset();
    assert(this->objects == 1);
    b.reset();
    assert(this->objects == 0);
  }

  _CCCL_HOST_DEVICE void test_any_iempty_pointer_to_model()
  {
    static_assert(::cuda::std::regular<cuda::__basic_any<iempty<>*>>);
    static_assert(sizeof(cuda::__basic_any<iempty<>*>) == 2 * sizeof(void*));

    assert(cuda::__basic_any<iempty<>*>{} == nullptr);
    assert(nullptr == cuda::__basic_any<iempty<>*>{});

    immovable im;
    cuda::__basic_any<iempty<>*> a = &im;
    assert(a.has_value() == true);
    assert(a.type() == _CCCL_TYPEID(immovable*));
    assert(a.interface() == _CCCL_TYPEID(iempty<>));
    assert(a.__in_situ() == true);

    [[maybe_unused]]
    immovable** p = cuda::__any_cast<immovable*>(&a);
    assert((p && *p == &im));

    int i    = 42;
    int*& pi = a.emplace<int*>(&i);
    assert(a.has_value() == true);
    assert(a.type() == _CCCL_TYPEID(int*));
    assert(a.interface() == _CCCL_TYPEID(iempty<>));
    assert(a.__in_situ() == true);
    assert(a == a);
    assert(a != nullptr);
    assert(nullptr != a);
    assert(pi == &i);

    int* const* ppi = cuda::__any_cast<int*>(&::cuda::std::as_const(a));
    assert(&pi == ppi);

    cuda::__basic_any<iempty<> const*> b = a;
    assert(b);
    assert(cuda::__any_cast<int const*>(&b));
    assert(&i == *cuda::__any_cast<int const*>(&b));
    assert(a == b);
    assert(b == a);
    b = nullptr;
    assert(a);
    assert(!b);
    assert(a != b);
    assert(b != a);

    int const k                                                     = 0;
    [[maybe_unused]] cuda::__basic_any<cuda::__imovable<> const*> c = &k;
    static_assert(!::cuda::std::constructible_from<cuda::__basic_any<cuda::__imovable<>>, decltype(*c)>);
  }

  _CCCL_HOST_DEVICE void test_any_ibase_pointer_to_model()
  {
    Foo<IsSmall> foo{42, this};
    cuda::__basic_any<ibase<>*> a = &foo;
    assert(a.has_value() == true);
    assert(a.type() == _CCCL_TYPEID(Foo<IsSmall>*));
    assert(a.interface() == _CCCL_TYPEID(ibase<>));
    assert(a.__in_situ() == true);
    assert(this->objects == 1);
    assert(a->foo(1) == 43);

    Foo<IsSmall> foo2{43, this};
    a.emplace<Foo<IsSmall>*>(&foo2);
    assert(this->objects == 2);
    assert(a->foo(2) == 45);

    a.reset();
    assert(a.has_value() == false);
    assert(a.type() == _CCCL_TYPEID(void));
    assert(a.interface() == _CCCL_TYPEID(ibase<>));
    assert(this->objects == 2);
  }

  _CCCL_HOST_DEVICE void test_any_pointers_from_derived_to_base_conversions()
  {
    static_assert(::cuda::std::constructible_from<cuda::__basic_any<ibase<>*>, cuda::__basic_any<iderived<>*>>);
    static_assert(::cuda::std::constructible_from<cuda::__basic_any<ibase<> const*>, cuda::__basic_any<iderived<>*>>);
    static_assert(
      ::cuda::std::constructible_from<cuda::__basic_any<iderived<> const*>, cuda::__basic_any<iderived<>*>>);

    static_assert(!::cuda::std::constructible_from<cuda::__basic_any<iderived<>*>, cuda::__basic_any<ibase<>*>>);
    static_assert(!::cuda::std::constructible_from<cuda::__basic_any<iderived<>*>, cuda::__basic_any<ibase<> const*>>);
    static_assert(
      !::cuda::std::constructible_from<cuda::__basic_any<iderived<>*>, cuda::__basic_any<iderived<> const*>>);
    static_assert(!::cuda::std::constructible_from<cuda::__basic_any<iderived<>*>, cuda::__basic_any<iderived<>*>*>);

    static_assert(!::cuda::std::convertible_to<cuda::__basic_any<ibase<>*>, cuda::__basic_any<iderived<>*>>);
    static_assert(!::cuda::std::convertible_to<cuda::__basic_any<ibase<> const*>, cuda::__basic_any<iderived<>*>>);
    static_assert(!::cuda::std::convertible_to<cuda::__basic_any<iderived<> const*>, cuda::__basic_any<iderived<>*>>);

    static_assert(::cuda::std::convertible_to<cuda::__basic_any<iderived<>*>, cuda::__basic_any<ibase<>*>>);
    static_assert(::cuda::std::convertible_to<cuda::__basic_any<iderived<>*>, cuda::__basic_any<ibase<> const*>>);
    static_assert(::cuda::std::convertible_to<cuda::__basic_any<iderived<>*>, cuda::__basic_any<iderived<> const*>>);
    static_assert(!::cuda::std::constructible_from<cuda::__basic_any<iderived<>*>*, cuda::__basic_any<iderived<>*>>);

    Bar<IsSmall> bar{42, this};
    cuda::__basic_any<iderived<>*> a{&bar};
    assert(a.has_value() == true);
    assert(a.type() == _CCCL_TYPEID(Bar<IsSmall>*));
    assert(a.interface() == _CCCL_TYPEID(iderived<>));
    assert(a.__in_situ() == true);
    assert(this->objects == 1);
    assert(a->foo(2) == 44);
    assert(a->bar(2) == 84);
    assert(cuda::__any_cast<Bar<IsSmall>*>(&a));
    assert((*cuda::__any_cast<Bar<IsSmall>*>(&a))->j == 42);

    // construct a base any from a derived any:
    cuda::__basic_any<ibase<>*> b{a};
    assert(b.has_value() == true);
    assert(b.type() == _CCCL_TYPEID(Bar<IsSmall>*));
    assert(b.interface() == _CCCL_TYPEID(iderived<>));
    assert(b.__in_situ() == true);
    assert(this->objects == 1);
    assert(b->foo(1) == 43);
    assert(cuda::__any_cast<Bar<IsSmall>*>(&b));
    assert((*cuda::__any_cast<Bar<IsSmall>*>(&b))->j == 42);

    Bar<IsSmall> bar2{-1, this};
    a.emplace<Bar<IsSmall>*>(&bar2);
    assert(a->foo(2) == 1);
    b = ::cuda::std::move(a);
    assert(b->foo(2) == 1);
    assert(cuda::__any_cast<Bar<IsSmall>*>(&b));
    assert((*cuda::__any_cast<Bar<IsSmall>*>(&b))->j == -1);

    a.reset();
    assert(this->objects == 2);
    b.reset();
    assert(this->objects == 2);
  }

  _CCCL_HOST_DEVICE void test_any_value_pointer_interop()
  {
    cuda::__basic_any<iderived<>> a{::cuda::std::in_place_type<Bar<IsSmall>>, 42, this};
    assert(a.__in_situ() == IsSmall);
    assert(this->objects == 1);

    cuda::__basic_any<iderived<>*> pa1 = &a;
    assert(pa1);
    assert(pa1 == &a);
    assert(&a == pa1);
    assert(!(pa1 != &a));
    assert(!(&a != pa1));
    assert(pa1->bar(2) == 84);

    Bar<IsSmall> bar{-1, this};
    cuda::__basic_any<ibase<> const*> pb2 = &bar;
    assert(pb2 == &bar);
    assert(&bar == pb2);
    assert(!(pb2 != &bar));
    assert(!(&bar != pb2));
    assert(this->objects == 2);

    cuda::__basic_any<iderived<> const*> pb3 = &bar;
    a                                        = *pb3;
    assert(a.has_value() == true);
    assert(a.type() == _CCCL_TYPEID(Bar<IsSmall>));
    assert(a.interface() == _CCCL_TYPEID(iderived<>));
    assert(a.__in_situ() == IsSmall);
    assert(this->objects == 2);
    assert(a.foo(2) == 1);

    bar.j = -2;
    a     = *pb3;
    assert(a.foo(2) == 0);
    assert(bar.j == -2);

    bar.j = 10;
    a     = ::cuda::std::move(*pb3);
    assert(a.foo(2) == 12);
    assert(bar.j == 10);

    cuda::__basic_any<iderived<>*> pb4 = &bar;
    bar.j                              = 20;
    a                                  = ::cuda::std::move(*pb4);
    assert(a.foo(2) == 22);
    assert(bar.j == INT_MAX); // bar is moved from
  }

  _CCCL_HOST_DEVICE void test_cuda_basic_any_references()
  {
    Bar<IsSmall> bar{42, this};
    cuda::__basic_any<iderived<>&> a{bar};
    assert(a.has_value() == true);
    assert(a.type() == _CCCL_TYPEID(Bar<IsSmall>));
    assert(a.interface() == _CCCL_TYPEID(iderived<>));
    assert(a.__in_situ() == true);
    assert(cuda::__any_cast<Bar<IsSmall>>(&a) == &bar);
    assert(this->objects == 1);

    cuda::__basic_any<ibase<>&> b = a;
    assert(b.has_value() == true);
    assert(b.type() == _CCCL_TYPEID(Bar<IsSmall>));
    assert(b.interface() == _CCCL_TYPEID(iderived<>));
    assert(b.__in_situ() == true);
    assert(this->objects == 1);

    cuda::__basic_any<iderived<>*> pa = &a;
    assert(pa == &bar);

    cuda::__basic_any<iderived<>> c = a; // should copy the referenced object
    assert(c.has_value() == true);
    assert(c.type() == _CCCL_TYPEID(Bar<IsSmall>));
    assert(c.interface() == _CCCL_TYPEID(iderived<>));
    assert(c.__in_situ() == IsSmall);
    assert(this->objects == 2);
    assert(cuda::__any_cast<Bar<IsSmall>>(&c));
    assert(cuda::__any_cast<Bar<IsSmall>>(&c) != &bar);
    assert(cuda::__any_cast<Bar<IsSmall>>(&c)->j == 42);

    bar.j = -1;
    c     = a; // should copy the referenced object
    assert(this->objects == 2);
    assert(cuda::__any_cast<Bar<IsSmall>>(&c)->j == -1);
    assert(a.bar(2) == -2);

    static_assert(
      ::cuda::std::constructible_from<cuda::__basic_any<cuda::__imovable<>>, cuda::__basic_any<cuda::__imovable<>>>);
    static_assert(
      !::cuda::std::constructible_from<cuda::__basic_any<cuda::__imovable<>>, cuda::__basic_any<cuda::__imovable<>&>>);
    static_assert(!::cuda::std::constructible_from<cuda::__basic_any<cuda::__imovable<>>,
                                                   cuda::__basic_any<cuda::__imovable<> const&>>);

    static_assert(
      !::cuda::std::constructible_from<cuda::__basic_any<cuda::__imovable<>>, cuda::__basic_any<cuda::__imovable<>*>>);

    static_assert(::cuda::std::constructible_from<cuda::__basic_any<cuda::__imovable<> const&>,
                                                  cuda::__basic_any<cuda::__imovable<>>>);
    static_assert(::cuda::std::constructible_from<cuda::__basic_any<cuda::__imovable<> const&>,
                                                  cuda::__basic_any<cuda::__imovable<>>&>);
  }

  struct cast_to_derived
  {
    template <class _Tp>
    _CCCL_HOST_DEVICE auto operator()(_Tp&& arg) const
      -> decltype(cuda::__dynamic_any_cast<iderived<>>(static_cast<_Tp&&>(arg)));
  };

  _CCCL_HOST_DEVICE void test_cuda_dynamic_any_cast()
  {
    static_assert(!::cuda::std::__is_callable_v<cast_to_derived, cuda::__basic_any<ibase<>>&>);
    static_assert(!::cuda::std::__is_callable_v<cast_to_derived, cuda::__basic_any<ibase<>&>>);
    static_assert(::cuda::std::__is_callable_v<cast_to_derived, cuda::__basic_any<cuda::__ireference<ibase<>>>>);
    static_assert(!::cuda::std::__is_callable_v<cast_to_derived, cuda::__basic_any<cuda::__ireference<ibase<>>>&>);

    // dynamic cast to a value
    cuda::__basic_any<iderived<>> a{::cuda::std::in_place_type<Bar<IsSmall>>, 42, this};
    cuda::__basic_any<ibase<>> b = a;
    auto c                       = cuda::__dynamic_any_cast<iderived<>>(::cuda::std::move(b));
    assert(c.has_value());
    assert(a.bar(2) == c.bar(2));
    assert(cuda::__any_cast<Bar<IsSmall>>(&a) != cuda::__any_cast<Bar<IsSmall>>(&c));

    cuda::__basic_any<ibase<>*> pa = &a;
    auto d                         = cuda::__dynamic_any_cast<iderived<>>(::cuda::std::move(*pa));
    assert(d.has_value());
    assert(cuda::__any_cast<Bar<IsSmall>>(&a)->j == INT_MAX); // moved from
    assert(cuda::__any_cast<Bar<IsSmall>>(&d)->j == 42);

    // basic_anyer
    cuda::__basic_any<ibase<>*> pb = &d;
    auto pd                        = cuda::__dynamic_any_cast<iderived<>*>(pb);
    assert(pd != nullptr);
    assert(cuda::__any_cast<Bar<IsSmall>>(&*pd)->j == 42);
  }

  _CCCL_HOST_DEVICE void test_equality_comparable()
  {
    cuda::__basic_any<iregular<>> a{42};
    cuda::__basic_any<iregular<>> b{42};
    cuda::__basic_any<iregular<>> c{43};
    cuda::__basic_any<iregular<>> d{::cuda::std::in_place_type<Regular>, 42};

    assert(a == a);
    assert(a == b);
    assert(!(a == c));
    assert(!(a == d));
    assert(!(a != a));
    assert(!(a != b));
    assert(a != c);
    assert(a != d);

    cuda::__basic_any<iregular<> const&> e = a;
    assert(e == e);
    assert(e == a);
    assert(e == b);
    assert(!(e == c));
    assert(!(e == d));
    assert(!(e != e));
    assert(!(e != b));
    assert(e != c);
    assert(e != d);

    assert(a == 42);
    assert(42 == a);
    assert(a != 43);
    assert(43 != a);
  }

  _CCCL_HOST_DEVICE void test_basic_any_test_for_ambiguous_conversions()
  {
    int i = 42;
    any_regular_ref ref{i};

    any_regular a = ref;
    a             = ref;
  }
};

template <class TestType>
_CCCL_HOST_DEVICE void test_basic_any()
{
  BasicAnyTest<TestType> test;
  test.test_type_traits();
  test.test_empty_interface_can_hold_anything();
  test.test_interface_with_one_member_function();
  test.test_single_interface_extension();
  test.test_any_iempty_pointer_to_model();
  test.test_any_ibase_pointer_to_model();
  test.test_any_pointers_from_derived_to_base_conversions();
  test.test_any_value_pointer_interop();
  test.test_cuda_basic_any_references();
  test.test_cuda_dynamic_any_cast();
  test.test_equality_comparable();
  test.test_basic_any_test_for_ambiguous_conversions();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_basic_any<SmallType>(); test_basic_any<LargeType>();))
  return 0;
}
