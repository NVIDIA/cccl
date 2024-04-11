//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class In, class Out>
// concept indirectly_copyable_storable;

#include <cuda/std/iterator>

#include "MoveOnly.h"
#include "test_macros.h"

struct CopyOnly
{
  CopyOnly(CopyOnly&&)                 = delete;
  CopyOnly(CopyOnly const&)            = default;
  CopyOnly& operator=(CopyOnly&&)      = delete;
  CopyOnly& operator=(CopyOnly const&) = default;
  CopyOnly()                           = default;
};

template <class T>
struct PointerTo
{
  using value_type = T;
  __host__ __device__ T& operator*() const;
};

// Copying the underlying object between pointers (or dereferenceable classes) works. This is a non-exhaustive check
// because this functionality comes from `indirectly_copyable`.
static_assert(cuda::std::indirectly_copyable_storable<int*, int*>);
static_assert(cuda::std::indirectly_copyable_storable<const int*, int*>);
static_assert(!cuda::std::indirectly_copyable_storable<int*, const int*>);
static_assert(!cuda::std::indirectly_copyable_storable<const int*, const int*>);
static_assert(cuda::std::indirectly_copyable_storable<int*, int[2]>);
static_assert(!cuda::std::indirectly_copyable_storable<int[2], int*>);
static_assert(!cuda::std::indirectly_copyable_storable<MoveOnly*, MoveOnly*>);
static_assert(!cuda::std::indirectly_copyable_storable<PointerTo<MoveOnly>, PointerTo<MoveOnly>>);
// `indirectly_copyable_storable` requires the type to be `copyable`, which in turns requires it to be `movable`.
static_assert(!cuda::std::indirectly_copyable_storable<CopyOnly*, CopyOnly*>);
static_assert(!cuda::std::indirectly_copyable_storable<PointerTo<CopyOnly>, PointerTo<CopyOnly>>);

// The dereference operator returns a different type from `value_type` and the reference type cannot be assigned from a
// non-const lvalue of `ValueType` (but all other forms of assignment from `ValueType` work).
struct NoLvalueAssignment
{
  struct ValueType;

  struct ReferenceType
  {
    __host__ __device__ ReferenceType& operator=(ValueType const&);
    ReferenceType& operator=(ValueType&) = delete;
    __host__ __device__ ReferenceType& operator=(ValueType&&);
    __host__ __device__ ReferenceType& operator=(ValueType const&&);
  };

  struct ValueType
  {
    __host__ __device__ operator ReferenceType&() const;
  };

  using value_type = ValueType;
  __host__ __device__ ReferenceType& operator*() const;
};

static_assert(cuda::std::indirectly_writable<NoLvalueAssignment, cuda::std::iter_reference_t<NoLvalueAssignment>>);
static_assert(!cuda::std::indirectly_writable<NoLvalueAssignment, cuda::std::iter_value_t<NoLvalueAssignment>&>);
static_assert(cuda::std::indirectly_writable<NoLvalueAssignment, const cuda::std::iter_value_t<NoLvalueAssignment>&>);
static_assert(cuda::std::indirectly_writable<NoLvalueAssignment, cuda::std::iter_value_t<NoLvalueAssignment>&&>);
static_assert(cuda::std::indirectly_writable<NoLvalueAssignment, const cuda::std::iter_value_t<NoLvalueAssignment>&&>);
static_assert(!cuda::std::indirectly_copyable_storable<NoLvalueAssignment, NoLvalueAssignment>);

// The dereference operator returns a different type from `value_type` and the reference type cannot be assigned from a
// const lvalue of `ValueType` (but all other forms of assignment from `ValueType` work).
struct NoConstLvalueAssignment
{
  struct ValueType;

  struct ReferenceType
  {
    ReferenceType& operator=(ValueType const&) = delete;
    __host__ __device__ ReferenceType& operator=(ValueType&);
    __host__ __device__ ReferenceType& operator=(ValueType&&);
    __host__ __device__ ReferenceType& operator=(ValueType const&&);
  };

  struct ValueType
  {
    __host__ __device__ operator ReferenceType&() const;
  };

  using value_type = ValueType;
  __host__ __device__ ReferenceType& operator*() const;
};

static_assert(
  cuda::std::indirectly_writable<NoConstLvalueAssignment, cuda::std::iter_reference_t<NoConstLvalueAssignment>>);
static_assert(
  cuda::std::indirectly_writable<NoConstLvalueAssignment, cuda::std::iter_value_t<NoConstLvalueAssignment>&>);
static_assert(
  !cuda::std::indirectly_writable<NoConstLvalueAssignment, const cuda::std::iter_value_t<NoConstLvalueAssignment>&>);
static_assert(
  cuda::std::indirectly_writable<NoConstLvalueAssignment, cuda::std::iter_value_t<NoConstLvalueAssignment>&&>);
static_assert(
  cuda::std::indirectly_writable<NoConstLvalueAssignment, const cuda::std::iter_value_t<NoConstLvalueAssignment>&&>);
static_assert(!cuda::std::indirectly_copyable_storable<NoConstLvalueAssignment, NoConstLvalueAssignment>);

// The dereference operator returns a different type from `value_type` and the reference type cannot be assigned from a
// non-const rvalue of `ValueType` (but all other forms of assignment from `ValueType` work).
struct NoRvalueAssignment
{
  struct ValueType;

  struct ReferenceType
  {
    __host__ __device__ ReferenceType& operator=(ValueType const&);
    __host__ __device__ ReferenceType& operator=(ValueType&);
    ReferenceType& operator=(ValueType&&) = delete;
    __host__ __device__ ReferenceType& operator=(ValueType const&&);
  };

  struct ValueType
  {
    __host__ __device__ operator ReferenceType&() const;
  };

  using value_type = ValueType;
  __host__ __device__ ReferenceType& operator*() const;
};

static_assert(cuda::std::indirectly_writable<NoRvalueAssignment, cuda::std::iter_reference_t<NoRvalueAssignment>>);
static_assert(cuda::std::indirectly_writable<NoRvalueAssignment, cuda::std::iter_value_t<NoRvalueAssignment>&>);
static_assert(cuda::std::indirectly_writable<NoRvalueAssignment, const cuda::std::iter_value_t<NoRvalueAssignment>&>);
static_assert(!cuda::std::indirectly_writable<NoRvalueAssignment, cuda::std::iter_value_t<NoRvalueAssignment>&&>);
static_assert(cuda::std::indirectly_writable<NoRvalueAssignment, const cuda::std::iter_value_t<NoRvalueAssignment>&&>);
static_assert(!cuda::std::indirectly_copyable_storable<NoRvalueAssignment, NoRvalueAssignment>);

// The dereference operator returns a different type from `value_type` and the reference type cannot be assigned from a
// const rvalue of `ValueType` (but all other forms of assignment from `ValueType` work).
struct NoConstRvalueAssignment
{
  struct ValueType;

  struct ReferenceType
  {
    __host__ __device__ ReferenceType& operator=(ValueType const&);
    __host__ __device__ ReferenceType& operator=(ValueType&);
    __host__ __device__ ReferenceType& operator=(ValueType&&);
    ReferenceType& operator=(ValueType const&&) = delete;
  };

  struct ValueType
  {
    __host__ __device__ operator ReferenceType&() const;
  };

  using value_type = ValueType;
  __host__ __device__ ReferenceType& operator*() const;
};

static_assert(
  cuda::std::indirectly_writable<NoConstRvalueAssignment, cuda::std::iter_reference_t<NoConstRvalueAssignment>>);
static_assert(
  cuda::std::indirectly_writable<NoConstRvalueAssignment, cuda::std::iter_value_t<NoConstRvalueAssignment>&>);
static_assert(
  cuda::std::indirectly_writable<NoConstRvalueAssignment, const cuda::std::iter_value_t<NoConstRvalueAssignment>&>);
static_assert(
  cuda::std::indirectly_writable<NoConstRvalueAssignment, cuda::std::iter_value_t<NoConstRvalueAssignment>&&>);
static_assert(
  !cuda::std::indirectly_writable<NoConstRvalueAssignment, const cuda::std::iter_value_t<NoConstRvalueAssignment>&&>);
static_assert(!cuda::std::indirectly_copyable_storable<NoConstRvalueAssignment, NoConstRvalueAssignment>);

struct DeletedCopyCtor
{
  DeletedCopyCtor(DeletedCopyCtor const&)            = delete;
  DeletedCopyCtor& operator=(DeletedCopyCtor const&) = default;
};

#if TEST_STD_VER > 2017 || !defined(TEST_COMPILER_MSVC) //  multiple versions of a defaulted special member functions
                                                        //  are not allowed
struct DeletedNonconstCopyCtor
{
  DeletedNonconstCopyCtor(DeletedNonconstCopyCtor const&)            = default;
  DeletedNonconstCopyCtor(DeletedNonconstCopyCtor&)                  = delete;
  DeletedNonconstCopyCtor& operator=(DeletedNonconstCopyCtor const&) = default;
};
static_assert(!cuda::std::indirectly_copyable_storable<DeletedNonconstCopyCtor*, DeletedNonconstCopyCtor*>);
#endif // TEST_STD_VER > 2017 || !defined(TEST_COMPILER_MSVC)

struct DeletedMoveCtor
{
  DeletedMoveCtor(DeletedMoveCtor&&)            = delete;
  DeletedMoveCtor& operator=(DeletedMoveCtor&&) = default;
};

#if TEST_STD_VER > 2017 || !defined(TEST_COMPILER_MSVC) //  multiple versions of a defaulted special member functions
                                                        //  are not allowed
struct DeletedConstMoveCtor
{
  DeletedConstMoveCtor(DeletedConstMoveCtor&&)            = default;
  DeletedConstMoveCtor(DeletedConstMoveCtor const&&)      = delete;
  DeletedConstMoveCtor& operator=(DeletedConstMoveCtor&&) = default;
};
static_assert(!cuda::std::indirectly_copyable_storable<DeletedConstMoveCtor*, DeletedConstMoveCtor*>);
#endif // TEST_STD_VER > 2017 || !defined(TEST_COMPILER_MSVC)

struct DeletedCopyAssignment
{
  DeletedCopyAssignment(DeletedCopyAssignment const&)            = default;
  DeletedCopyAssignment& operator=(DeletedCopyAssignment const&) = delete;
};

#if TEST_STD_VER > 2017 || !defined(TEST_COMPILER_MSVC) //  multiple versions of a defaulted special member functions
                                                        //  are not allowed
struct DeletedNonconstCopyAssignment
{
  DeletedNonconstCopyAssignment(DeletedNonconstCopyAssignment const&)            = default;
  DeletedNonconstCopyAssignment& operator=(DeletedNonconstCopyAssignment const&) = default;
  DeletedNonconstCopyAssignment& operator=(DeletedNonconstCopyAssignment&)       = delete;
};
static_assert(!cuda::std::indirectly_copyable_storable<DeletedNonconstCopyAssignment*, DeletedNonconstCopyAssignment*>);
#endif // TEST_STD_VER > 2017 || !defined(TEST_COMPILER_MSVC)

struct DeletedMoveAssignment
{
  DeletedMoveAssignment(DeletedMoveAssignment&&)            = default;
  DeletedMoveAssignment& operator=(DeletedMoveAssignment&&) = delete;
};

struct DeletedConstMoveAssignment
{
  DeletedConstMoveAssignment(DeletedConstMoveAssignment&&)            = default;
  DeletedConstMoveAssignment& operator=(DeletedConstMoveAssignment&&) = delete;
};

static_assert(!cuda::std::indirectly_copyable_storable<DeletedCopyCtor*, DeletedCopyCtor*>);
static_assert(!cuda::std::indirectly_copyable_storable<DeletedMoveCtor*, DeletedMoveCtor*>);
static_assert(!cuda::std::indirectly_copyable_storable<DeletedCopyAssignment*, DeletedCopyAssignment*>);
static_assert(!cuda::std::indirectly_copyable_storable<DeletedMoveAssignment*, DeletedMoveAssignment*>);
static_assert(!cuda::std::indirectly_copyable_storable<DeletedConstMoveAssignment*, DeletedConstMoveAssignment*>);

struct InconsistentIterator
{
  struct ValueType;

  struct ReferenceType
  {
    __host__ __device__ ReferenceType& operator=(ValueType const&);
  };

  struct ValueType
  {
    ValueType() = default;
    __host__ __device__ ValueType(const ReferenceType&);
  };

  using value_type = ValueType;
  __host__ __device__ ReferenceType& operator*() const;
};

// `ValueType` can be constructed with a `ReferenceType` and assigned to a `ReferenceType`, so it does model
// `indirectly_copyable_storable`.
static_assert(cuda::std::indirectly_copyable_storable<InconsistentIterator, InconsistentIterator>);

struct CommonType
{};

// ReferenceType is a (proxy) reference for ValueType, but ValueType is not constructible from ReferenceType.
struct NotConstructibleFromRefIn
{
  struct ReferenceType;

  struct ValueType
  {
    ValueType(ReferenceType) = delete;
    __host__ __device__ operator CommonType&() const;
  };

  struct ReferenceType
  {
    __host__ __device__ operator CommonType&() const;
  };

  using value_type = ValueType;
  __host__ __device__ ReferenceType& operator*() const;
};

template <template <class> class X, template <class> class Y>
struct cuda::std::
  basic_common_reference<NotConstructibleFromRefIn::ValueType, NotConstructibleFromRefIn::ReferenceType, X, Y>
{
  using type = CommonType&;
};

template <template <class> class X, template <class> class Y>
struct cuda::std::
  basic_common_reference<NotConstructibleFromRefIn::ReferenceType, NotConstructibleFromRefIn::ValueType, X, Y>
{
  using type = CommonType&;
};

static_assert(
  cuda::std::common_reference_with<NotConstructibleFromRefIn::ValueType&, NotConstructibleFromRefIn::ReferenceType&>);

struct AssignableFromAnything
{
  template <class T>
  __host__ __device__ AssignableFromAnything& operator=(T&&);
};

// A type that can't be constructed from its own reference isn't `indirectly_copyable_storable`, even when assigning it
// to a type that can be assigned from anything.
static_assert(!cuda::std::indirectly_copyable_storable<NotConstructibleFromRefIn, AssignableFromAnything*>);

// ReferenceType is a (proxy) reference for ValueType, but ValueType is not assignable from ReferenceType.
struct NotAssignableFromRefIn
{
  struct ReferenceType;

  struct ValueType
  {
    __host__ __device__ ValueType(ReferenceType);
    ValueType& operator=(ReferenceType) = delete;
    __host__ __device__ operator CommonType&() const;
  };

  struct ReferenceType
  {
    __host__ __device__ operator CommonType&() const;
  };

  using value_type = ValueType;
  __host__ __device__ ReferenceType& operator*() const;
};

template <template <class> class X, template <class> class Y>
struct cuda::std::basic_common_reference<NotAssignableFromRefIn::ValueType, NotAssignableFromRefIn::ReferenceType, X, Y>
{
  using type = CommonType&;
};

template <template <class> class X, template <class> class Y>
struct cuda::std::basic_common_reference<NotAssignableFromRefIn::ReferenceType, NotAssignableFromRefIn::ValueType, X, Y>
{
  using type = CommonType&;
};

static_assert(
  cuda::std::common_reference_with<NotAssignableFromRefIn::ValueType&, NotAssignableFromRefIn::ReferenceType&>);

// A type that can't be assigned from its own reference isn't `indirectly_copyable_storable`, even when assigning it
// to a type that can be assigned from anything.
static_assert(!cuda::std::indirectly_copyable_storable<NotAssignableFromRefIn, AssignableFromAnything*>);

int main(int, char**)
{
  return 0;
}
