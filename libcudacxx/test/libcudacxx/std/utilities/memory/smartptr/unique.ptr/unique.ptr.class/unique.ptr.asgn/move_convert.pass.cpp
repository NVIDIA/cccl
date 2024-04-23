//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03
// UNSUPPORTED: nvrtc

// <memory>

// unique_ptr

// Test unique_ptr converting move ctor

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "type_id.h"
#include "unique_ptr_test_helper.h"

template <int ID = 0>
struct GenericDeleter
{
  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(void*) const {}
};

template <int ID = 0>
struct GenericConvertingDeleter
{
  template <int OID>
  __host__ __device__ TEST_CONSTEXPR_CXX23 GenericConvertingDeleter(GenericConvertingDeleter<OID>)
  {}

  template <int OID>
  __host__ __device__ TEST_CONSTEXPR_CXX23 GenericConvertingDeleter& operator=(GenericConvertingDeleter<OID> const&)
  {
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(void*) const {}
};

template <class T, class U>
using EnableIfNotSame = typename cuda::std::enable_if<
  !cuda::std::is_same<typename cuda::std::decay<T>::type, typename cuda::std::decay<U>::type>::value>::type;

template <class Templ, class Other>
struct is_specialization;

template <template <int> class Templ, int ID1, class Other>
struct is_specialization<Templ<ID1>, Other> : cuda::std::false_type
{};

template <template <int> class Templ, int ID1, int ID2>
struct is_specialization<Templ<ID1>, Templ<ID2>> : cuda::std::true_type
{};

template <class Templ, class Other>
using EnableIfSpecialization =
  typename cuda::std::enable_if<is_specialization<Templ, typename cuda::std::decay<Other>::type>::value>::type;

template <int ID>
struct TrackingDeleter;
template <int ID>
struct ConstTrackingDeleter;

template <int ID>
struct TrackingDeleter
{
  __host__ __device__ TrackingDeleter()
      : arg_type(&makeArgumentID<>())
  {}

  __host__ __device__ TrackingDeleter(TrackingDeleter const&)
      : arg_type(&makeArgumentID<TrackingDeleter const&>())
  {}

  __host__ __device__ TrackingDeleter(TrackingDeleter&&)
      : arg_type(&makeArgumentID<TrackingDeleter&&>())
  {}

  template <class T, class = EnableIfSpecialization<TrackingDeleter, T>>
  __host__ __device__ TrackingDeleter(T&&)
      : arg_type(&makeArgumentID<T&&>())
  {}

  __host__ __device__ TrackingDeleter& operator=(TrackingDeleter const&)
  {
    arg_type = &makeArgumentID<TrackingDeleter const&>();
    return *this;
  }

  __host__ __device__ TrackingDeleter& operator=(TrackingDeleter&&)
  {
    arg_type = &makeArgumentID<TrackingDeleter&&>();
    return *this;
  }

  template <class T, class = EnableIfSpecialization<TrackingDeleter, T>>
  __host__ __device__ TrackingDeleter& operator=(T&&)
  {
    arg_type = &makeArgumentID<T&&>();
    return *this;
  }

  __host__ __device__ void operator()(void*) const {}

public:
  __host__ __device__ TypeID const* reset() const
  {
    TypeID const* tmp = arg_type;
    arg_type          = nullptr;
    return tmp;
  }

  mutable TypeID const* arg_type;
};

template <int ID>
struct ConstTrackingDeleter
{
  __host__ __device__ ConstTrackingDeleter()
      : arg_type(&makeArgumentID<>())
  {}

  __host__ __device__ ConstTrackingDeleter(ConstTrackingDeleter const&)
      : arg_type(&makeArgumentID<ConstTrackingDeleter const&>())
  {}

  __host__ __device__ ConstTrackingDeleter(ConstTrackingDeleter&&)
      : arg_type(&makeArgumentID<ConstTrackingDeleter&&>())
  {}

  template <class T, class = EnableIfSpecialization<ConstTrackingDeleter, T>>
  __host__ __device__ ConstTrackingDeleter(T&&)
      : arg_type(&makeArgumentID<T&&>())
  {}

  __host__ __device__ const ConstTrackingDeleter& operator=(ConstTrackingDeleter const&) const
  {
    arg_type = &makeArgumentID<ConstTrackingDeleter const&>();
    return *this;
  }

  __host__ __device__ const ConstTrackingDeleter& operator=(ConstTrackingDeleter&&) const
  {
    arg_type = &makeArgumentID<ConstTrackingDeleter&&>();
    return *this;
  }

  template <class T, class = EnableIfSpecialization<ConstTrackingDeleter, T>>
  __host__ __device__ const ConstTrackingDeleter& operator=(T&&) const
  {
    arg_type = &makeArgumentID<T&&>();
    return *this;
  }

  __host__ __device__ void operator()(void*) const {}

public:
  __host__ __device__ TypeID const* reset() const
  {
    TypeID const* tmp = arg_type;
    arg_type          = nullptr;
    return tmp;
  }

  mutable TypeID const* arg_type;
};

template <class ExpectT, int ID>
__host__ __device__ bool checkArg(TrackingDeleter<ID> const& d)
{
  return d.arg_type && *d.arg_type == makeArgumentID<ExpectT>();
}

template <class ExpectT, int ID>
__host__ __device__ bool checkArg(ConstTrackingDeleter<ID> const& d)
{
  return d.arg_type && *d.arg_type == makeArgumentID<ExpectT>();
}

template <class From, bool AssignIsConst = false>
struct AssignDeleter
{
  TEST_CONSTEXPR_CXX23 AssignDeleter()                     = default;
  TEST_CONSTEXPR_CXX23 AssignDeleter(AssignDeleter const&) = default;
  TEST_CONSTEXPR_CXX23 AssignDeleter(AssignDeleter&&)      = default;

  AssignDeleter& operator=(AssignDeleter const&) = delete;
  AssignDeleter& operator=(AssignDeleter&&)      = delete;

  template <class T>
  AssignDeleter& operator=(T&&) && = delete;
  template <class T>
  AssignDeleter& operator=(T&&) const&& = delete;

  template <class T, class = typename cuda::std::enable_if<cuda::std::is_same<T&&, From>::value && !AssignIsConst>::type>
  __host__ __device__ TEST_CONSTEXPR_CXX23 AssignDeleter& operator=(T&&) &
  {
    return *this;
  }

  template <class T, class = typename cuda::std::enable_if<cuda::std::is_same<T&&, From>::value && AssignIsConst>::type>
  __host__ __device__ TEST_CONSTEXPR_CXX23 const AssignDeleter& operator=(T&&) const&
  {
    return *this;
  }

  template <class T>
  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(T) const
  {}
};

template <class VT, class DDest, class DSource>
__host__ __device__ TEST_CONSTEXPR_CXX23 void doDeleterTest()
{
  using U1 = cuda::std::unique_ptr<VT, DDest>;
  using U2 = cuda::std::unique_ptr<VT, DSource>;
  static_assert(cuda::std::is_nothrow_assignable<U1, U2&&>::value, "");
  typename cuda::std::decay<DDest>::type ddest;
  typename cuda::std::decay<DSource>::type dsource;
  U1 u1(nullptr, ddest);
  U2 u2(nullptr, dsource);
  u1 = cuda::std::move(u2);
}

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_sfinae()
{
  typedef typename cuda::std::conditional<IsArray, A[], A>::type VT;

  { // Test that different non-reference deleter types are allowed so long
    // as they convert to each other.
    using U1 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0>>;
    using U2 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<1>>;
    static_assert(cuda::std::is_assignable<U1, U2&&>::value, "");
  }
  { // Test that different non-reference deleter types are disallowed when
    // they cannot convert.
    using U1 = cuda::std::unique_ptr<VT, GenericDeleter<0>>;
    using U2 = cuda::std::unique_ptr<VT, GenericDeleter<1>>;
    static_assert(!cuda::std::is_assignable<U1, U2&&>::value, "");
  }
  { // Test that if the deleter assignment is not valid the assignment operator
    // SFINAEs.
    using U1 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0> const&>;
    using U2 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0>>;
    using U3 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0>&>;
    using U4 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<1>>;
    using U5 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<1> const&>;
    static_assert(!cuda::std::is_assignable<U1, U2&&>::value, "");
    static_assert(!cuda::std::is_assignable<U1, U3&&>::value, "");
    static_assert(!cuda::std::is_assignable<U1, U4&&>::value, "");
    static_assert(!cuda::std::is_assignable<U1, U5&&>::value, "");

    using U1C = cuda::std::unique_ptr<const VT, GenericConvertingDeleter<0> const&>;
    static_assert(cuda::std::is_nothrow_assignable<U1C, U1&&>::value, "");
  }
  { // Test that if the deleter assignment is not valid the assignment operator
    // SFINAEs.
    using U1 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0>&>;
    using U2 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0>>;
    using U3 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0>&>;
    using U4 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<1>>;
    using U5 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<1> const&>;

    static_assert(cuda::std::is_nothrow_assignable<U1, U2&&>::value, "");
    static_assert(cuda::std::is_nothrow_assignable<U1, U3&&>::value, "");
    static_assert(cuda::std::is_nothrow_assignable<U1, U4&&>::value, "");
    static_assert(cuda::std::is_nothrow_assignable<U1, U5&&>::value, "");

    using U1C = cuda::std::unique_ptr<const VT, GenericConvertingDeleter<0>&>;
    static_assert(cuda::std::is_nothrow_assignable<U1C, U1&&>::value, "");
  }
  { // Test that non-reference destination deleters can be assigned
    // from any source deleter type with a suitable conversion. Including
    // reference types.
    using U1 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0>>;
    using U2 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0>&>;
    using U3 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<0> const&>;
    using U4 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<1>>;
    using U5 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<1>&>;
    using U6 = cuda::std::unique_ptr<VT, GenericConvertingDeleter<1> const&>;
    static_assert(cuda::std::is_assignable<U1, U2&&>::value, "");
    static_assert(cuda::std::is_assignable<U1, U3&&>::value, "");
    static_assert(cuda::std::is_assignable<U1, U4&&>::value, "");
    static_assert(cuda::std::is_assignable<U1, U5&&>::value, "");
    static_assert(cuda::std::is_assignable<U1, U6&&>::value, "");
  }
  /////////////////////////////////////////////////////////////////////////////
  {
    using Del = GenericDeleter<0>;
    using AD  = AssignDeleter<Del&&>;
    using ADC = AssignDeleter<Del&&, /*AllowConstAssign*/ true>;
    doDeleterTest<VT, AD, Del>();
    doDeleterTest<VT, AD&, Del>();
    doDeleterTest<VT, ADC const&, Del>();
  }
  {
    using Del = GenericDeleter<0>;
    using AD  = AssignDeleter<Del&>;
    using ADC = AssignDeleter<Del&, /*AllowConstAssign*/ true>;
    doDeleterTest<VT, AD, Del&>();
    doDeleterTest<VT, AD&, Del&>();
    doDeleterTest<VT, ADC const&, Del&>();
  }
  {
    using Del = GenericDeleter<0>;
    using AD  = AssignDeleter<Del const&>;
    using ADC = AssignDeleter<Del const&, /*AllowConstAssign*/ true>;
    doDeleterTest<VT, AD, Del const&>();
    doDeleterTest<VT, AD&, Del const&>();
    doDeleterTest<VT, ADC const&, Del const&>();
  }
}

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_noexcept()
{
  typedef typename cuda::std::conditional<IsArray, A[], A>::type VT;
  {
    typedef cuda::std::unique_ptr<const VT> APtr;
    typedef cuda::std::unique_ptr<VT> BPtr;
    static_assert(cuda::std::is_nothrow_assignable<APtr, BPtr>::value, "");
  }
  {
    typedef cuda::std::unique_ptr<const VT, CDeleter<const VT>> APtr;
    typedef cuda::std::unique_ptr<VT, CDeleter<VT>> BPtr;
    static_assert(cuda::std::is_nothrow_assignable<APtr, BPtr>::value, "");
  }
  {
    typedef cuda::std::unique_ptr<const VT, NCDeleter<const VT>&> APtr;
    typedef cuda::std::unique_ptr<VT, NCDeleter<const VT>&> BPtr;
    static_assert(cuda::std::is_nothrow_assignable<APtr, BPtr>::value, "");
  }
  {
    typedef cuda::std::unique_ptr<const VT, const NCConstDeleter<const VT>&> APtr;
    typedef cuda::std::unique_ptr<VT, const NCConstDeleter<const VT>&> BPtr;
    static_assert(cuda::std::is_nothrow_assignable<APtr, BPtr>::value, "");
  }
}

template <bool IsArray>
__host__ __device__ void test_deleter_value_category()
{
  typedef typename cuda::std::conditional<IsArray, A[], A>::type VT;
  using TD1 = TrackingDeleter<1>;
  using TD2 = TrackingDeleter<2>;
  TD1 d1;
  TD2 d2;
  using CD1 = ConstTrackingDeleter<1>;
  using CD2 = ConstTrackingDeleter<2>;
  CD1 cd1;
  CD2 cd2;

  { // Test non-reference deleter conversions
    using U1 = cuda::std::unique_ptr<VT, TD1>;
    using U2 = cuda::std::unique_ptr<VT, TD2>;
    U1 u1;
    U2 u2;
    u1.get_deleter().reset();
    u1 = cuda::std::move(u2);
    assert(checkArg<TD2&&>(u1.get_deleter()));
  }
  { // Test assignment to non-const ref
    using U1 = cuda::std::unique_ptr<VT, TD1&>;
    using U2 = cuda::std::unique_ptr<VT, TD2>;
    U1 u1(nullptr, d1);
    U2 u2;
    u1.get_deleter().reset();
    u1 = cuda::std::move(u2);
    assert(checkArg<TD2&&>(u1.get_deleter()));
  }
  { // Test assignment to const&.
    using U1 = cuda::std::unique_ptr<VT, CD1 const&>;
    using U2 = cuda::std::unique_ptr<VT, CD2>;
    U1 u1(nullptr, cd1);
    U2 u2;
    u1.get_deleter().reset();
    u1 = cuda::std::move(u2);
    assert(checkArg<CD2&&>(u1.get_deleter()));
  }

  { // Test assignment from non-const ref
    using U1 = cuda::std::unique_ptr<VT, TD1>;
    using U2 = cuda::std::unique_ptr<VT, TD2&>;
    U1 u1;
    U2 u2(nullptr, d2);
    u1.get_deleter().reset();
    u1 = cuda::std::move(u2);
    assert(checkArg<TD2&>(u1.get_deleter()));
  }
  { // Test assignment from const ref
    using U1 = cuda::std::unique_ptr<VT, TD1>;
    using U2 = cuda::std::unique_ptr<VT, TD2 const&>;
    U1 u1;
    U2 u2(nullptr, d2);
    u1.get_deleter().reset();
    u1 = cuda::std::move(u2);
    assert(checkArg<TD2 const&>(u1.get_deleter()));
  }

  { // Test assignment from non-const ref
    using U1 = cuda::std::unique_ptr<VT, TD1&>;
    using U2 = cuda::std::unique_ptr<VT, TD2&>;
    U1 u1(nullptr, d1);
    U2 u2(nullptr, d2);
    u1.get_deleter().reset();
    u1 = cuda::std::move(u2);
    assert(checkArg<TD2&>(u1.get_deleter()));
  }
  { // Test assignment from const ref
    using U1 = cuda::std::unique_ptr<VT, TD1&>;
    using U2 = cuda::std::unique_ptr<VT, TD2 const&>;
    U1 u1(nullptr, d1);
    U2 u2(nullptr, d2);
    u1.get_deleter().reset();
    u1 = cuda::std::move(u2);
    assert(checkArg<TD2 const&>(u1.get_deleter()));
  }

  { // Test assignment from non-const ref
    using U1 = cuda::std::unique_ptr<VT, CD1 const&>;
    using U2 = cuda::std::unique_ptr<VT, CD2&>;
    U1 u1(nullptr, cd1);
    U2 u2(nullptr, cd2);
    u1.get_deleter().reset();
    u1 = cuda::std::move(u2);
    assert(checkArg<CD2&>(u1.get_deleter()));
  }
  { // Test assignment from const ref
    using U1 = cuda::std::unique_ptr<VT, CD1 const&>;
    using U2 = cuda::std::unique_ptr<VT, CD2 const&>;
    U1 u1(nullptr, cd1);
    U2 u2(nullptr, cd2);
    u1.get_deleter().reset();
    u1 = cuda::std::move(u2);
    assert(checkArg<CD2 const&>(u1.get_deleter()));
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  {
    test_sfinae</*IsArray*/ false>();
    test_noexcept<false>();
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      test_deleter_value_category<false>();
    }
  }
  {
    test_sfinae</*IsArray*/ true>();
    test_noexcept<true>();
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      test_deleter_value_category<true>();
    }
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2023
  static_assert(test());
#endif // TEST_STD_VER >= 2023

  return 0;
}
