//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>

#include <cuda/experimental/__utility/basic_any.cuh>

#include <testing.cuh>

#undef interface

using immovable = cudax::detail::__immovable;

struct TestCounters
{
  int objects = 0;
};

template <class TestType>
struct BasicAnyTestsFixture : TestCounters
{
  BasicAnyTestsFixture() {}
};

template <class...>
struct iempty : cudax::interface<iempty>
{};

static_assert(cudax::extension_of<iempty<>, iempty<>>);
static_assert(cudax::extension_of<iempty<>, cudax::iunknown>);

template <class...>
struct ibase : cudax::interface<ibase, cudax::extends<cudax::imovable<>>>
{
  int foo(int i)
  {
    return cudax::virtcall<&ibase::foo>(this, i);
  }

  template <class T>
  using overrides = cudax::overrides_for<T, &T::foo>;
};

template <class...>
struct iderived : cudax::interface<iderived, cudax::extends<ibase<>, cudax::icopyable<>>>
{
  int bar(int i)
  {
    return cudax::virtcall<&iderived::bar>(this, i);
  }

  template <class T>
  using overrides = cudax::overrides_for<T, &T::bar>;
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
  _CUDA_VSTD_NOVERSION::byte buffer[cudax::__default_buffer_size]{};
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
struct iregular : cudax::interface<iregular, cudax::extends<cudax::icopyable<>, cudax::iequality_comparable<>>>
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

TEMPLATE_TEST_CASE_METHOD(BasicAnyTestsFixture, "basic_any tests", "[utility][basic_any]", SmallType, LargeType)
{
  constexpr bool IsSmall = _CUDA_VSTD::is_same_v<TestType, SmallType>;

  SECTION("type traits")
  {
    STATIC_REQUIRE(_CUDA_VSTD::is_standard_layout_v<cudax::basic_any<iregular<>>>);
    STATIC_REQUIRE(_CUDA_VSTD::is_standard_layout_v<cudax::basic_any<iregular<>*>>);
    STATIC_REQUIRE(_CUDA_VSTD::is_standard_layout_v<cudax::basic_any<iregular<>&>>);
  }

  SECTION("empty interface can hold anything")
  {
    STATIC_REQUIRE_FALSE(_CUDA_VSTD::move_constructible<cudax::basic_any<iempty<>>>);
    cudax::basic_any<iempty<>> a{42};
    CHECK(a.has_value() == true);
    CHECK(a.type() == _CCCL_TYPEID(int));
    CHECK(a.interface() == _CCCL_TYPEID(iempty<>));
    REQUIRE(cudax::any_cast<int>(&a));
    CHECK(cudax::any_cast<int>(&a) == cudax::any_cast<void>(&a));
    CHECK(*cudax::any_cast<int>(&a) == 42);

    a.emplace<immovable>();
    CHECK(a.has_value() == true);
    CHECK(a.type() == _CCCL_TYPEID(immovable));
    CHECK(a.interface() == _CCCL_TYPEID(iempty<>));

    a.reset();
    CHECK(a.has_value() == false);
    CHECK(a.type() == _CCCL_TYPEID(void));
    CHECK(a.interface() == _CCCL_TYPEID(iempty<>));
  }

  SECTION("interface with one member function")
  {
    STATIC_REQUIRE(_CUDA_VSTD::move_constructible<cudax::basic_any<ibase<>>>);
    STATIC_REQUIRE_FALSE(_CUDA_VSTD::copy_constructible<cudax::basic_any<ibase<>>>);

    cudax::basic_any<ibase<>> a{_CUDA_VSTD::in_place_type<Foo<IsSmall>>, 42, this};
    CHECK(a.has_value() == true);
    CHECK(a.type() == _CCCL_TYPEID(Foo<IsSmall>));
    CHECK(a.interface() == _CCCL_TYPEID(ibase<>));
    CHECK(a.__in_situ() == IsSmall);
    CHECK(this->objects == 1);
    CHECK(a.foo(1) == 43);

    a.emplace<Foo<IsSmall>>(43, this);
    CHECK(a.foo(2) == 45);
    CHECK(this->objects == 1);

    a.reset();
    CHECK(a.has_value() == false);
    CHECK(a.type() == _CCCL_TYPEID(void));
    CHECK(a.interface() == _CCCL_TYPEID(ibase<>));
    CHECK(this->objects == 0);
  }

  SECTION("single interface extension")
  {
    STATIC_REQUIRE(_CUDA_VSTD::move_constructible<cudax::basic_any<iderived<>>>);
    STATIC_REQUIRE(_CUDA_VSTD::copy_constructible<cudax::basic_any<iderived<>>>);

    cudax::basic_any<iderived<>> a{_CUDA_VSTD::in_place_type<Bar<IsSmall>>, 42, this};
    CHECK(a.has_value() == true);
    CHECK(a.type() == _CCCL_TYPEID(Bar<IsSmall>));
    CHECK(a.interface() == _CCCL_TYPEID(iderived<>));
    CHECK(a.__in_situ() == IsSmall);
    CHECK(this->objects == 1);
    CHECK(a.foo(2) == 44);
    CHECK(a.bar(2) == 84);
    CHECK(cudax::any_cast<Bar<IsSmall>>(&a)->j == 42);

    // construct a base any from a derived any:
    cudax::basic_any<ibase<>> b{_CUDA_VSTD::move(a)};
    CHECK(b.has_value() == true);
    CHECK(b.type() == _CCCL_TYPEID(Bar<IsSmall>));
    CHECK(b.interface() == _CCCL_TYPEID(iderived<>));
    CHECK(b.__in_situ() == IsSmall);
    CHECK(this->objects == 1);
    CHECK(b.foo(1) == 43);
    CHECK(cudax::any_cast<Bar<IsSmall>>(&b)->j == 42);

    a.emplace<Bar<IsSmall>>(-1, this);
    CHECK(a.foo(2) == 1);
    b = _CUDA_VSTD::move(a);
    CHECK(b.foo(2) == 1);
    CHECK(cudax::any_cast<Bar<IsSmall>>(&b)->j == -1);

    a.reset();
    CHECK(this->objects == 1);
    b.reset();
    CHECK(this->objects == 0);
  }

  SECTION("any iempty pointer to model")
  {
    STATIC_REQUIRE(_CUDA_VSTD::regular<cudax::basic_any<iempty<>*>>);
    STATIC_REQUIRE(sizeof(cudax::basic_any<iempty<>*>) == 2 * sizeof(void*));

    CHECK(cudax::basic_any<iempty<>*>{} == nullptr);
    CHECK(nullptr == cudax::basic_any<iempty<>*>{});

    immovable im;
    cudax::basic_any<iempty<>*> a = &im;
    CHECK(a.has_value() == true);
    CHECK(a.type() == _CCCL_TYPEID(immovable*));
    CHECK(a.interface() == _CCCL_TYPEID(iempty<>));
    CHECK(a.__in_situ() == true);

    [[maybe_unused]]
    immovable** p = cudax::any_cast<immovable*>(&a);
    CHECK((p && *p == &im));

    int i    = 42;
    int*& pi = a.emplace<int*>(&i);
    CHECK(a.has_value() == true);
    CHECK(a.type() == _CCCL_TYPEID(int*));
    CHECK(a.interface() == _CCCL_TYPEID(iempty<>));
    CHECK(a.__in_situ() == true);
    CHECK(a == a);
    CHECK(a != nullptr);
    CHECK(nullptr != a);
    CHECK(pi == &i);

    int* const* ppi = cudax::any_cast<int*>(&_CUDA_VSTD::as_const(a));
    CHECK(&pi == ppi);

    cudax::basic_any<iempty<> const*> b = a;
    CHECK(b);
    REQUIRE(cudax::any_cast<int const*>(&b));
    CHECK(&i == *cudax::any_cast<int const*>(&b));
    CHECK(a == b);
    CHECK(b == a);
    b = nullptr;
    CHECK(a);
    CHECK(!b);
    CHECK(a != b);
    CHECK(b != a);

    int const k                                                   = 0;
    [[maybe_unused]] cudax::basic_any<cudax::imovable<> const*> c = &k;
    STATIC_REQUIRE(!_CUDA_VSTD::constructible_from<cudax::basic_any<cudax::imovable<>>, decltype(*c)>);
  }

  SECTION("any ibase pointer to model")
  {
    Foo<IsSmall> foo{42, this};
    cudax::basic_any<ibase<>*> a = &foo;
    CHECK(a.has_value() == true);
    CHECK(a.type() == _CCCL_TYPEID(Foo<IsSmall>*));
    CHECK(a.interface() == _CCCL_TYPEID(ibase<>));
    CHECK(a.__in_situ() == true);
    CHECK(this->objects == 1);
    CHECK(a->foo(1) == 43);

    Foo<IsSmall> foo2{43, this};
    a.emplace<Foo<IsSmall>*>(&foo2);
    CHECK(this->objects == 2);
    CHECK(a->foo(2) == 45);

    a.reset();
    CHECK(a.has_value() == false);
    CHECK(a.type() == _CCCL_TYPEID(void));
    CHECK(a.interface() == _CCCL_TYPEID(ibase<>));
    CHECK(this->objects == 2);
  }

  SECTION("any pointers from derived to base conversions")
  {
    STATIC_REQUIRE(_CUDA_VSTD::constructible_from<cudax::basic_any<ibase<>*>, cudax::basic_any<iderived<>*>>);
    STATIC_REQUIRE(_CUDA_VSTD::constructible_from<cudax::basic_any<ibase<> const*>, cudax::basic_any<iderived<>*>>);
    STATIC_REQUIRE(_CUDA_VSTD::constructible_from<cudax::basic_any<iderived<> const*>, cudax::basic_any<iderived<>*>>);

    STATIC_REQUIRE(!_CUDA_VSTD::constructible_from<cudax::basic_any<iderived<>*>, cudax::basic_any<ibase<>*>>);
    STATIC_REQUIRE(!_CUDA_VSTD::constructible_from<cudax::basic_any<iderived<>*>, cudax::basic_any<ibase<> const*>>);
    STATIC_REQUIRE(!_CUDA_VSTD::constructible_from<cudax::basic_any<iderived<>*>, cudax::basic_any<iderived<> const*>>);
    STATIC_REQUIRE(!_CUDA_VSTD::constructible_from<cudax::basic_any<iderived<>*>, cudax::basic_any<iderived<>*>*>);

    STATIC_REQUIRE(!_CUDA_VSTD::convertible_to<cudax::basic_any<ibase<>*>, cudax::basic_any<iderived<>*>>);
    STATIC_REQUIRE(!_CUDA_VSTD::convertible_to<cudax::basic_any<ibase<> const*>, cudax::basic_any<iderived<>*>>);
    STATIC_REQUIRE(!_CUDA_VSTD::convertible_to<cudax::basic_any<iderived<> const*>, cudax::basic_any<iderived<>*>>);

    STATIC_REQUIRE(_CUDA_VSTD::convertible_to<cudax::basic_any<iderived<>*>, cudax::basic_any<ibase<>*>>);
    STATIC_REQUIRE(_CUDA_VSTD::convertible_to<cudax::basic_any<iderived<>*>, cudax::basic_any<ibase<> const*>>);
    STATIC_REQUIRE(_CUDA_VSTD::convertible_to<cudax::basic_any<iderived<>*>, cudax::basic_any<iderived<> const*>>);
    STATIC_REQUIRE(!_CUDA_VSTD::constructible_from<cudax::basic_any<iderived<>*>*, cudax::basic_any<iderived<>*>>);

    Bar<IsSmall> bar{42, this};
    cudax::basic_any<iderived<>*> a{&bar};
    CHECK(a.has_value() == true);
    CHECK(a.type() == _CCCL_TYPEID(Bar<IsSmall>*));
    CHECK(a.interface() == _CCCL_TYPEID(iderived<>));
    CHECK(a.__in_situ() == true);
    CHECK(this->objects == 1);
    CHECK(a->foo(2) == 44);
    CHECK(a->bar(2) == 84);
    REQUIRE(cudax::any_cast<Bar<IsSmall>*>(&a));
    CHECK((*cudax::any_cast<Bar<IsSmall>*>(&a))->j == 42);

    // construct a base any from a derived any:
    cudax::basic_any<ibase<>*> b{a};
    CHECK(b.has_value() == true);
    CHECK(b.type() == _CCCL_TYPEID(Bar<IsSmall>*));
    CHECK(b.interface() == _CCCL_TYPEID(iderived<>));
    CHECK(b.__in_situ() == true);
    CHECK(this->objects == 1);
    CHECK(b->foo(1) == 43);
    REQUIRE(cudax::any_cast<Bar<IsSmall>*>(&b));
    CHECK((*cudax::any_cast<Bar<IsSmall>*>(&b))->j == 42);

    Bar<IsSmall> bar2{-1, this};
    a.emplace<Bar<IsSmall>*>(&bar2);
    CHECK(a->foo(2) == 1);
    b = _CUDA_VSTD::move(a);
    CHECK(b->foo(2) == 1);
    REQUIRE(cudax::any_cast<Bar<IsSmall>*>(&b));
    CHECK((*cudax::any_cast<Bar<IsSmall>*>(&b))->j == -1);

    a.reset();
    CHECK(this->objects == 2);
    b.reset();
    CHECK(this->objects == 2);
  }

  SECTION("any value/pointer interop")
  {
    cudax::basic_any<iderived<>> a{_CUDA_VSTD::in_place_type<Bar<IsSmall>>, 42, this};
    CHECK(a.__in_situ() == IsSmall);
    CHECK(this->objects == 1);

    cudax::basic_any<iderived<>*> pa1 = &a;
    CHECK(pa1);
    CHECK(pa1 == &a);
    CHECK(&a == pa1);
    CHECK_FALSE(pa1 != &a);
    CHECK_FALSE(&a != pa1);
    CHECK(pa1->bar(2) == 84);

    Bar<IsSmall> bar{-1, this};
    cudax::basic_any<ibase<> const*> pb2 = &bar;
    CHECK(pb2 == &bar);
    CHECK(&bar == pb2);
    CHECK_FALSE(pb2 != &bar);
    CHECK_FALSE(&bar != pb2);
    CHECK(this->objects == 2);

    cudax::basic_any<iderived<> const*> pb3 = &bar;
    a                                       = *pb3;
    CHECK(a.has_value() == true);
    CHECK(a.type() == _CCCL_TYPEID(Bar<IsSmall>));
    CHECK(a.interface() == _CCCL_TYPEID(iderived<>));
    CHECK(a.__in_situ() == IsSmall);
    CHECK(this->objects == 2);
    CHECK(a.foo(2) == 1);

    bar.j = -2;
    a     = *pb3;
    CHECK(a.foo(2) == 0);
    CHECK(bar.j == -2);

    bar.j = 10;
    a     = _CUDA_VSTD::move(*pb3);
    CHECK(a.foo(2) == 12);
    CHECK(bar.j == 10);

    cudax::basic_any<iderived<>*> pb4 = &bar;
    bar.j                             = 20;
    a                                 = _CUDA_VSTD::move(*pb4);
    CHECK(a.foo(2) == 22);
    CHECK(bar.j == INT_MAX); // bar is moved from
  }

  SECTION("cudax::basic_any references")
  {
    Bar<IsSmall> bar{42, this};
    cudax::basic_any<iderived<>&> a{bar};
    CHECK(a.has_value() == true);
    CHECK(a.type() == _CCCL_TYPEID(Bar<IsSmall>));
    CHECK(a.interface() == _CCCL_TYPEID(iderived<>));
    CHECK(a.__in_situ() == true);
    CHECK(cudax::any_cast<Bar<IsSmall>>(&a) == &bar);
    CHECK(this->objects == 1);

    cudax::basic_any<ibase<>&> b = a;
    CHECK(b.has_value() == true);
    CHECK(b.type() == _CCCL_TYPEID(Bar<IsSmall>));
    CHECK(b.interface() == _CCCL_TYPEID(iderived<>));
    CHECK(b.__in_situ() == true);
    CHECK(this->objects == 1);

    cudax::basic_any<iderived<>*> pa = &a;
    CHECK(pa == &bar);

    cudax::basic_any<iderived<>> c = a; // should copy the referenced object
    CHECK(c.has_value() == true);
    CHECK(c.type() == _CCCL_TYPEID(Bar<IsSmall>));
    CHECK(c.interface() == _CCCL_TYPEID(iderived<>));
    CHECK(c.__in_situ() == IsSmall);
    CHECK(this->objects == 2);
    REQUIRE(cudax::any_cast<Bar<IsSmall>>(&c));
    CHECK(cudax::any_cast<Bar<IsSmall>>(&c) != &bar);
    CHECK(cudax::any_cast<Bar<IsSmall>>(&c)->j == 42);

    bar.j = -1;
    c     = a; // should copy the referenced object
    CHECK(this->objects == 2);
    CHECK(cudax::any_cast<Bar<IsSmall>>(&c)->j == -1);
    CHECK(a.bar(2) == -2);

    STATIC_REQUIRE(
      _CUDA_VSTD::constructible_from<cudax::basic_any<cudax::imovable<>>, cudax::basic_any<cudax::imovable<>>>);
    STATIC_REQUIRE(
      !_CUDA_VSTD::constructible_from<cudax::basic_any<cudax::imovable<>>, cudax::basic_any<cudax::imovable<>&>>);
    STATIC_REQUIRE(
      !_CUDA_VSTD::constructible_from<cudax::basic_any<cudax::imovable<>>, cudax::basic_any<cudax::imovable<> const&>>);

    STATIC_REQUIRE(
      !_CUDA_VSTD::constructible_from<cudax::basic_any<cudax::imovable<>>, cudax::basic_any<cudax::imovable<>*>>);

    STATIC_REQUIRE(
      _CUDA_VSTD::constructible_from<cudax::basic_any<cudax::imovable<> const&>, cudax::basic_any<cudax::imovable<>>>);
    STATIC_REQUIRE(
      _CUDA_VSTD::constructible_from<cudax::basic_any<cudax::imovable<> const&>, cudax::basic_any<cudax::imovable<>>&>);
  }

  SECTION("cudax::dynamic_any_cast")
  {
    auto cast_to_derived_fn =
      [](auto&& arg) -> decltype(cudax::dynamic_any_cast<iderived<>>(static_cast<decltype(arg)>(arg))) {
      throw;
    };
    using cast_to_derived = decltype(cast_to_derived_fn);

    STATIC_REQUIRE_FALSE(_CUDA_VSTD::__is_callable_v<cast_to_derived, cudax::basic_any<ibase<>>&>);
    STATIC_REQUIRE_FALSE(_CUDA_VSTD::__is_callable_v<cast_to_derived, cudax::basic_any<ibase<>&>>);
    STATIC_REQUIRE(_CUDA_VSTD::__is_callable_v<cast_to_derived, cudax::basic_any<cudax::__ireference<ibase<>>>>);
    STATIC_REQUIRE_FALSE(_CUDA_VSTD::__is_callable_v<cast_to_derived, cudax::basic_any<cudax::__ireference<ibase<>>>&>);

    // dynamic cast to a value
    cudax::basic_any<iderived<>> a{_CUDA_VSTD::in_place_type<Bar<IsSmall>>, 42, this};
    cudax::basic_any<ibase<>> b = a;
    auto c                      = cudax::dynamic_any_cast<iderived<>>(_CUDA_VSTD::move(b));
    CHECK(c.has_value());
    CHECK(a.bar(2) == c.bar(2));
    CHECK(cudax::any_cast<Bar<IsSmall>>(&a) != cudax::any_cast<Bar<IsSmall>>(&c));

    cudax::basic_any<ibase<>*> pa = &a;
    auto d                        = cudax::dynamic_any_cast<iderived<>>(_CUDA_VSTD::move(*pa));
    CHECK(d.has_value());
    CHECK(cudax::any_cast<Bar<IsSmall>>(&a)->j == INT_MAX); // moved from
    CHECK(cudax::any_cast<Bar<IsSmall>>(&d)->j == 42);

    // basic_anyer
    cudax::basic_any<ibase<>*> pb = &d;
    auto pd                       = cudax::dynamic_any_cast<iderived<>*>(pb);
    CHECK(pd != nullptr);
    CHECK(cudax::any_cast<Bar<IsSmall>>(&*pd)->j == 42);
  }

  SECTION("equality_comparable")
  {
    cudax::basic_any<iregular<>> a{42};
    cudax::basic_any<iregular<>> b{42};
    cudax::basic_any<iregular<>> c{43};
    cudax::basic_any<iregular<>> d{_CUDA_VSTD::in_place_type<Regular>, 42};

    CHECK(a == a);
    CHECK(a == b);
    CHECK_FALSE(a == c);
    CHECK_FALSE(a == d);
    CHECK_FALSE(a != a);
    CHECK_FALSE(a != b);
    CHECK(a != c);
    CHECK(a != d);

    cudax::basic_any<iregular<> const&> e = a;
    CHECK(e == e);
    CHECK(e == a);
    CHECK(e == b);
    CHECK_FALSE(e == c);
    CHECK_FALSE(e == d);
    CHECK_FALSE(e != e);
    CHECK_FALSE(e != b);
    CHECK(e != c);
    CHECK(e != d);

    CHECK(a == 42);
    CHECK(42 == a);
    CHECK(a != 43);
    CHECK(43 != a);
  }
}

struct any_regular : cudax::basic_any<iregular<>>
{
  using cudax::basic_any<iregular<>>::basic_any;
};

struct any_regular_ref : cudax::basic_any<iregular<>&>
{
  using cudax::basic_any<iregular<>&>::basic_any;
};

C2H_TEST("basic_any test for ambiguous conversions", "[utility][basic_any]")
{
  int i = 42;
  any_regular_ref ref{i};

  any_regular a = ref;
  a             = ref;
}
