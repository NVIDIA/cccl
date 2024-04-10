//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SUPPORT_TYPE_ID_H
#define SUPPORT_TYPE_ID_H

#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "test_macros.h"

#ifndef TEST_COMPILER_NVRTC
#  include <string>
#  include <typeinfo>

#  include "demangle.h"
#endif // TEST_COMPILER_NVRTC

// TypeID - Represent a unique identifier for a type. TypeID allows equality
// comparisons between different types.
struct TypeID
{
  __host__ __device__ friend bool operator==(TypeID const& LHS, TypeID const& RHS)
  {
    return LHS.m_id == RHS.m_id;
  }
  __host__ __device__ friend bool operator!=(TypeID const& LHS, TypeID const& RHS)
  {
    return LHS.m_id != RHS.m_id;
  }

#if 0
  std::string name() const {
    return demangle(m_id);
  }
#else
  __host__ __device__ const char* name() const
  {
    return m_id;
  }
#endif

  __host__ __device__ void dump() const
  {
#if 0
    std::string s = name();
    std::printf("TypeID: %s\n", s.c_str());
#else
    printf("TypeID: %s\n", m_id);
#endif
  }

private:
  __host__ __device__ explicit constexpr TypeID(const char* xid)
      : m_id(xid)
  {}

  TypeID(const TypeID&)            = delete;
  TypeID& operator=(TypeID const&) = delete;

  const char* const m_id;
  template <class T>
  __host__ __device__ friend TypeID const& makeTypeIDImp();
};

// makeTypeID - Return the TypeID for the specified type 'T'.
template <class T>
__host__ __device__ inline TypeID const& makeTypeIDImp()
{
#ifdef __CUDA_ARCH__
  __constant__ static const TypeID id{__PRETTY_FUNCTION__};
#elif defined(_MSC_VER)
  static const TypeID id(__FUNCDNAME__);
#else
  static const TypeID id(__PRETTY_FUNCTION__);
#endif

  return id;
}

template <class T>
struct TypeWrapper
{};

template <class T>
__host__ __device__ inline TypeID const& makeTypeID()
{
  return makeTypeIDImp<TypeWrapper<T>>();
}

template <class... Args>
struct ArgumentListID
{};

// makeArgumentID - Create and return a unique identifier for a given set
// of arguments.
template <class... Args>
__host__ __device__ inline TypeID const& makeArgumentID()
{
  return makeTypeIDImp<ArgumentListID<Args...>>();
}

// COMPARE_TYPEID(...) is a utility macro for generating diagnostics when
// two typeid's are expected to be equal
#define COMPARE_TYPEID(LHS, RHS) CompareTypeIDVerbose(#LHS, LHS, #RHS, RHS)

__host__ __device__ inline bool
CompareTypeIDVerbose(const char* LHSString, TypeID const* LHS, const char* RHSString, TypeID const* RHS)
{
  if (*LHS == *RHS)
  {
    return true;
  }
#if 0
  std::printf("TypeID's not equal:\n");
  std::printf("%s: %s\n----------\n%s: %s\n",
              LHSString, LHS->name().c_str(),
              RHSString, RHS->name().c_str());
#else
  printf("TypeID's not equal:\n");
  printf("%s: %s\n----------\n%s: %s\n", LHSString, LHS->name(), RHSString, RHS->name());
#endif
  return false;
}

#endif // SUPPORT_TYPE_ID_H
