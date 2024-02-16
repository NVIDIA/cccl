//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<class D>
//   requires is_class_v<D> && same_as<D, remove_cv_t<D>>
// class view_interface;

#include <cuda/std/ranges>

#include <cuda/std/cassert>
#include <cuda/std/utility>
#include "test_macros.h"
#include "test_iterators.h"

#if TEST_STD_VER > 2017
template<class T>
concept ValidViewInterfaceType = requires { typename cuda::std::ranges::view_interface<T>; };
#else
template<class T, class = void>
constexpr bool ValidViewInterfaceType = false;

template<class T>
constexpr bool ValidViewInterfaceType<T, cuda::std::void_t<cuda::std::ranges::view_interface<T>>> = true;
#endif

struct Empty { };

static_assert(!ValidViewInterfaceType<void>);
static_assert(!ValidViewInterfaceType<void*>);
static_assert(!ValidViewInterfaceType<Empty*>);
static_assert(!ValidViewInterfaceType<Empty const>);
static_assert(!ValidViewInterfaceType<Empty &>);
static_assert( ValidViewInterfaceType<Empty>);

using InputIter = cpp20_input_iterator<const int*>;

struct InputRange : cuda::std::ranges::view_interface<InputRange> {
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  TEST_HOST_DEVICE constexpr InputIter begin() const { return InputIter(buff); }
  TEST_HOST_DEVICE constexpr InputIter end() const { return InputIter(buff + 8); }
};

struct NotSizedSentinel {
  using value_type = int;
  using difference_type = cuda::std::ptrdiff_t;
  using iterator_concept = cuda::std::forward_iterator_tag;

  explicit NotSizedSentinel() = default;
  TEST_HOST_DEVICE explicit NotSizedSentinel(int*);
  TEST_HOST_DEVICE int& operator*() const;
  TEST_HOST_DEVICE NotSizedSentinel& operator++();
  TEST_HOST_DEVICE NotSizedSentinel operator++(int);
  TEST_HOST_DEVICE bool operator==(NotSizedSentinel const&) const;
#if TEST_STD_VER < 2020
  TEST_HOST_DEVICE bool operator!=(NotSizedSentinel const&) const;
#endif
};
static_assert(cuda::std::forward_iterator<NotSizedSentinel>);

using ForwardIter = forward_iterator<int*>;

// So that we conform to sized_sentinel_for.
TEST_HOST_DEVICE constexpr cuda::std::ptrdiff_t operator-(const ForwardIter& x, const ForwardIter& y) {
    return base(x) - base(y);
}

struct ForwardRange : cuda::std::ranges::view_interface<ForwardRange> {
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  TEST_HOST_DEVICE constexpr ForwardIter begin() const { return ForwardIter(const_cast<int*>(buff)); }
  TEST_HOST_DEVICE constexpr ForwardIter end() const { return ForwardIter(const_cast<int*>(buff) + 8); }
};
static_assert(cuda::std::ranges::view<ForwardRange>);

struct MoveOnlyForwardRange : cuda::std::ranges::view_interface<MoveOnlyForwardRange> {
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  MoveOnlyForwardRange(MoveOnlyForwardRange const&) = delete;
  MoveOnlyForwardRange(MoveOnlyForwardRange &&) = default;
  MoveOnlyForwardRange& operator=(MoveOnlyForwardRange &&) = default;
  MoveOnlyForwardRange() = default;
  TEST_HOST_DEVICE constexpr ForwardIter begin() const { return ForwardIter(const_cast<int*>(buff)); }
  TEST_HOST_DEVICE constexpr ForwardIter end() const { return ForwardIter(const_cast<int*>(buff) + 8); }
};
static_assert(cuda::std::ranges::view<MoveOnlyForwardRange>);

struct MI : cuda::std::ranges::view_interface<InputRange>,
            cuda::std::ranges::view_interface<MoveOnlyForwardRange> {
};
static_assert(!cuda::std::ranges::view<MI>);

struct EmptyIsTrue : cuda::std::ranges::view_interface<EmptyIsTrue> {
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  TEST_HOST_DEVICE constexpr ForwardIter begin() const { return ForwardIter(const_cast<int*>(buff)); }
  TEST_HOST_DEVICE constexpr ForwardIter end() const { return ForwardIter(const_cast<int*>(buff) + 8); }
  TEST_HOST_DEVICE constexpr bool empty() const { return true; }
};
static_assert(cuda::std::ranges::view<EmptyIsTrue>);

struct SizeIsTen : cuda::std::ranges::view_interface<SizeIsTen> {
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  TEST_HOST_DEVICE constexpr ForwardIter begin() const { return ForwardIter(const_cast<int*>(buff)); }
  TEST_HOST_DEVICE constexpr ForwardIter end() const { return ForwardIter(const_cast<int*>(buff) + 8); }
  TEST_HOST_DEVICE constexpr size_t size() const { return 10; }
};
static_assert(cuda::std::ranges::view<SizeIsTen>);

using RAIter = random_access_iterator<int*>;

struct RARange : cuda::std::ranges::view_interface<RARange> {
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  TEST_HOST_DEVICE constexpr RAIter begin() const { return RAIter(const_cast<int*>(buff)); }
  TEST_HOST_DEVICE constexpr RAIter end() const { return RAIter(const_cast<int*>(buff) + 8); }
};
static_assert(cuda::std::ranges::view<RARange>);

using ContIter = contiguous_iterator<const int*>;

struct ContRange : cuda::std::ranges::view_interface<ContRange> {
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  TEST_HOST_DEVICE constexpr ContIter begin() const { return ContIter(buff); }
  TEST_HOST_DEVICE constexpr ContIter end() const { return ContIter(buff + 8); }
};
static_assert(cuda::std::ranges::view<ContRange>);

struct DataIsNull : cuda::std::ranges::view_interface<DataIsNull> {
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  TEST_HOST_DEVICE constexpr ContIter begin() const { return ContIter(buff); }
  TEST_HOST_DEVICE constexpr ContIter end() const { return ContIter(buff + 8); }
  TEST_HOST_DEVICE constexpr const int *data() const { return nullptr; }
};
static_assert(cuda::std::ranges::view<DataIsNull>);

struct BoolConvertibleComparison : cuda::std::ranges::view_interface<BoolConvertibleComparison> {
  struct ResultType {
    bool value;
    TEST_HOST_DEVICE constexpr operator bool() const { return value; }
  };

  struct SentinelType {
    int *base_;
    explicit SentinelType() = default;
    TEST_HOST_DEVICE constexpr explicit SentinelType(int *base) : base_(base) {}
    TEST_HOST_DEVICE friend constexpr ResultType operator==(ForwardIter const& iter, SentinelType const& sent) noexcept { return {base(iter) == sent.base_}; }
    TEST_HOST_DEVICE friend constexpr ResultType operator==(SentinelType const& sent, ForwardIter const& iter) noexcept { return {base(iter) == sent.base_}; }
    TEST_HOST_DEVICE friend constexpr ResultType operator!=(ForwardIter const& iter, SentinelType const& sent) noexcept { return {base(iter) != sent.base_}; }
    TEST_HOST_DEVICE friend constexpr ResultType operator!=(SentinelType const& sent, ForwardIter const& iter) noexcept { return {base(iter) != sent.base_}; }
  };

  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  TEST_HOST_DEVICE constexpr ForwardIter begin() const { return ForwardIter(const_cast<int*>(buff)); }
  TEST_HOST_DEVICE constexpr SentinelType end() const { return SentinelType(const_cast<int*>(buff) + 8); }
};
static_assert(cuda::std::ranges::view<BoolConvertibleComparison>);

#if TEST_STD_VER > 2017
template<class T>
concept EmptyInvocable = requires (T const& obj) { obj.empty(); };

template<class T>
concept BoolOpInvocable = requires (T const& obj) { bool(obj); };
#else
template<class T, class = void>
constexpr bool EmptyInvocable = false;
template<class T>
constexpr bool EmptyInvocable<T, cuda::std::void_t<decltype(cuda::std::declval<T const&>().empty())>> = true;

template<class T, class = void>
constexpr bool BoolOpInvocable = false;
template<class T>
constexpr bool BoolOpInvocable<T, cuda::std::void_t<decltype(bool(cuda::std::declval<T const&>()))>> = true;
#endif

TEST_HOST_DEVICE constexpr bool testEmpty() {
  static_assert(!EmptyInvocable<InputRange>);
  static_assert( EmptyInvocable<ForwardRange>);

  static_assert(!BoolOpInvocable<InputRange>);
  static_assert( BoolOpInvocable<ForwardRange>);

  ForwardRange forwardRange;
  assert(!forwardRange.empty());
  assert(!static_cast<ForwardRange const&>(forwardRange).empty());

  assert(forwardRange);
  assert(static_cast<ForwardRange const&>(forwardRange));

  assert(!cuda::std::ranges::empty(forwardRange));
  assert(!cuda::std::ranges::empty(static_cast<ForwardRange const&>(forwardRange)));

  EmptyIsTrue emptyTrue{};
  assert(emptyTrue.empty());
  assert(static_cast<EmptyIsTrue const&>(emptyTrue).empty());
  assert(!emptyTrue.cuda::std::ranges::view_interface<EmptyIsTrue>::empty());

  assert(!emptyTrue);
  assert(!static_cast<EmptyIsTrue const&>(emptyTrue));
  assert(!emptyTrue.cuda::std::ranges::view_interface<EmptyIsTrue>::operator bool());

  assert(cuda::std::ranges::empty(emptyTrue));
  assert(cuda::std::ranges::empty(static_cast<EmptyIsTrue const&>(emptyTrue)));

  // Try calling empty on an rvalue.
  MoveOnlyForwardRange moveOnly{};
  assert(!cuda::std::move(moveOnly).empty());

  BoolConvertibleComparison boolConv{};
  // old GCC seems to fall over the noexcept clauses here
#if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 9) \
 && (!defined(TEST_COMPILER_MSVC))                 \
 && (!defined(TEST_COMPILER_ICC))
  ASSERT_NOT_NOEXCEPT(boolConv.empty());
#endif

  assert(!boolConv.empty());
  assert(!static_cast<const BoolConvertibleComparison&>(boolConv).empty());

  assert(boolConv);
  assert(static_cast<const BoolConvertibleComparison&>(boolConv));

  assert(!cuda::std::ranges::empty(boolConv));
  assert(!cuda::std::ranges::empty(static_cast<const BoolConvertibleComparison&>(boolConv)));

  return true;
}

#if TESTS_STD_VER > 17
template<class T>
concept DataInvocable = requires (T const& obj) { obj.data(); };
#else
template<class T, class = void>
constexpr bool DataInvocable = false;
template<class T>
constexpr bool DataInvocable<T, cuda::std::void_t<decltype(cuda::std::declval<T const&>().data())>> = true;
#endif

TEST_HOST_DEVICE constexpr bool testData() {
  static_assert(!DataInvocable<ForwardRange>);
  static_assert( DataInvocable<ContRange>);

  ContRange contiguous{};
  assert(contiguous.data() == contiguous.buff);
  assert(static_cast<ContRange const&>(contiguous).data() == contiguous.buff);

  assert(cuda::std::ranges::data(contiguous) == contiguous.buff);
  assert(cuda::std::ranges::data(static_cast<ContRange const&>(contiguous)) == contiguous.buff);

  DataIsNull dataNull{};
  assert(dataNull.data() == nullptr);
  assert(static_cast<DataIsNull const&>(dataNull).data() == nullptr);
  assert(dataNull.cuda::std::ranges::view_interface<DataIsNull>::data() == dataNull.buff);

  assert(cuda::std::ranges::data(dataNull) == nullptr);
  assert(cuda::std::ranges::data(static_cast<DataIsNull const&>(dataNull)) == nullptr);

  return true;
}

#if TEST_STD_VER > 2017
template<class T>
concept SizeInvocable = requires (T const& obj) { obj.size(); };
#else
template<class T, class = void>
constexpr bool SizeInvocable = false;
template<class T>
constexpr bool SizeInvocable<T, cuda::std::void_t<decltype(cuda::std::declval<T const&>().size())>> = true;
#endif

TEST_HOST_DEVICE constexpr bool testSize() {
  static_assert(!SizeInvocable<InputRange>);
  static_assert(!SizeInvocable<NotSizedSentinel>);
  static_assert( SizeInvocable<ForwardRange>);

  // Test the test.
  static_assert(cuda::std::same_as<decltype(cuda::std::declval<ForwardIter>() - cuda::std::declval<ForwardIter>()), cuda::std::ptrdiff_t>);
  using UnsignedSize = cuda::std::make_unsigned_t<cuda::std::ptrdiff_t>;
  using SignedSize = cuda::std::common_type_t<cuda::std::ptrdiff_t, cuda::std::make_signed_t<UnsignedSize>>;

  ForwardRange forwardRange{};
  assert(forwardRange.size() == 8);
  assert(static_cast<ForwardRange const&>(forwardRange).size() == 8);

  assert(cuda::std::ranges::size(forwardRange) == 8);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::size(cuda::std::declval<ForwardRange>())), UnsignedSize>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::ssize(cuda::std::declval<ForwardRange>())), SignedSize>);

  assert(cuda::std::ranges::size(static_cast<ForwardRange const&>(forwardRange)) == 8);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::size(cuda::std::declval<ForwardRange const>())), UnsignedSize>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::ssize(cuda::std::declval<ForwardRange const>())), SignedSize>);

  SizeIsTen sizeTen{};
  assert(sizeTen.size() == 10);
  assert(static_cast<SizeIsTen const&>(sizeTen).size() == 10);
  assert(sizeTen.cuda::std::ranges::view_interface<SizeIsTen>::size() == 8);

  assert(cuda::std::ranges::size(sizeTen) == 10);
  assert(cuda::std::ranges::size(static_cast<SizeIsTen const&>(sizeTen)) == 10);

  return true;
}

#if TEST_STD_VER > 2017
template<class T>
concept SubscriptInvocable = requires (T const& obj, size_t n) { obj[n]; };
#else
template<class T, class = void>
constexpr bool SubscriptInvocable = false;
template<class T>
constexpr bool SubscriptInvocable<T, cuda::std::void_t<decltype(cuda::std::declval<T const&>()[cuda::std::declval<size_t>()])>> = true;
#endif

TEST_HOST_DEVICE constexpr bool testSubscript() {
  static_assert(!SubscriptInvocable<ForwardRange>);
  static_assert( SubscriptInvocable<RARange>);

  RARange randomAccess{};
  assert(randomAccess[2] == 2);
  assert(static_cast<RARange const&>(randomAccess)[2] == 2);
  randomAccess[2] = 3;
  assert(randomAccess[2] == 3);

  return true;
}

#if TEST_STD_VER > 2017
template<class T>
concept FrontInvocable = requires (T const& obj) { obj.front(); };

template<class T>
concept BackInvocable = requires (T const& obj) { obj.back(); };
#else
template<class T, class = void>
constexpr bool FrontInvocable = false;
template<class T>
constexpr bool FrontInvocable<T, cuda::std::void_t<decltype(cuda::std::declval<T const&>().front())>> = true;
template<class T, class = void>

constexpr bool BackInvocable = false;
template<class T>
constexpr bool BackInvocable<T, cuda::std::void_t<decltype(cuda::std::declval<T const&>().back())>> = true;
#endif

TEST_HOST_DEVICE constexpr bool testFrontBack() {
  static_assert(!FrontInvocable<InputRange>);
  static_assert( FrontInvocable<ForwardRange>);
  static_assert(!BackInvocable<ForwardRange>);
  static_assert( BackInvocable<RARange>);

  ForwardRange forwardRange{};
  assert(forwardRange.front() == 0);
  assert(static_cast<ForwardRange const&>(forwardRange).front() == 0);
  forwardRange.front() = 2;
  assert(forwardRange.front() == 2);

  RARange randomAccess{};
  assert(randomAccess.front() == 0);
  assert(static_cast<RARange const&>(randomAccess).front() == 0);
  randomAccess.front() = 2;
  assert(randomAccess.front() == 2);

  assert(randomAccess.back() == 7);
  assert(static_cast<RARange const&>(randomAccess).back() == 7);
  randomAccess.back() = 2;
  assert(randomAccess.back() == 2);

  return true;
}

struct V1 : cuda::std::ranges::view_interface<V1> { };
struct V2 : cuda::std::ranges::view_interface<V2> { V1 base_; };
static_assert(sizeof(V2) == sizeof(V1));

int main(int, char**) {
  testEmpty();
  static_assert(testEmpty());

  testData();
  static_assert(testData());

  testSize();
  static_assert(testSize());

  testSubscript();
  static_assert(testSubscript());

  testFrontBack();
  static_assert(testFrontBack());

  return 0;
}
