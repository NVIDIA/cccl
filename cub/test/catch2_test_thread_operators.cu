// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/thread/thread_operators.cuh>

#include "test_util.h"
#include <c2h/catch2_test_helper.h>

template <class T>
T Make(int val)
{
  return T{val};
}

template <bool>
class BaseT
{
protected:
  int m_val{};

public:
  BaseT(int val)
      : m_val{val}
  {}
};

template <>
class BaseT<true>
{
protected:
  int m_val{};

public:
  BaseT(int val)
      : m_val{val}
  {}

  __host__ __device__ operator int() const
  {
    return m_val;
  }
};

#define CUSTOM_TYPE_FACTORY(NAME, RT, OP, CONVERTIBLE) \
  class Custom##NAME##T : public BaseT<CONVERTIBLE>    \
  {                                                    \
    explicit Custom##NAME##T(int val)                  \
        : BaseT<CONVERTIBLE>(val)                      \
    {}                                                 \
                                                       \
    friend Custom##NAME##T Make<Custom##NAME##T>(int); \
                                                       \
  public:                                              \
    __host__ __device__ RT operator OP(int val) const  \
    {                                                  \
      return m_val OP val;                             \
    }                                                  \
  }

CUSTOM_TYPE_FACTORY(Eq, bool, ==, false);

C2H_TEST("InequalityWrapper", "[thread_operator]")
{
  cuda::std::equal_to<> wrapped_op{};
  cub::InequalityWrapper<cuda::std::equal_to<>> op{wrapped_op};

  constexpr int const_magic_val = 42;
  int magic_val                 = const_magic_val;

  CHECK(op(const_magic_val, const_magic_val) == false);
  CHECK(op(const_magic_val, magic_val) == false);
  CHECK(op(const_magic_val, magic_val + 1) == true);

  CHECK(op(Make<CustomEqT>(magic_val), magic_val) == false);
  CHECK(op(Make<CustomEqT>(magic_val), magic_val + 1) == true);
}
