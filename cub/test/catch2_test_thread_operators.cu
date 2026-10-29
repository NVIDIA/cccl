// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/thread/thread_operators.cuh>

#include "cub_test_macros.h"
#include "test_util.h"

template <class T>
T Make(int val)
{
  return T{val};
}

template <bool>
struct BaseT
{
  int m_val{};
};

template <>
struct BaseT<true>
{
  int m_val{};

  __host__ __device__ operator int() const
  {
    return m_val;
  }
};

#define CUSTOM_TYPE_FACTORY(NAME, RT, OP, CONVERTIBLE) \
  struct Custom##NAME##T : BaseT<CONVERTIBLE>          \
  {                                                    \
    __host__ __device__ RT operator OP(int val) const  \
    {                                                  \
      return m_val OP val;                             \
    }                                                  \
  }

CUSTOM_TYPE_FACTORY(Eq, bool, ==, false);

CUB_TEST("InequalityWrapper", "[thread_operator]", CUB_SMALL)
{
  const cuda::std::equal_to<> wrapped_op{};
  cub::InequalityWrapper<cuda::std::equal_to<>> op{wrapped_op};

  constexpr int const_magic_val = 42;
  const int magic_val           = const_magic_val;

  CHECK(op(const_magic_val, const_magic_val) == false);
  CHECK(op(const_magic_val, magic_val) == false);
  CHECK(op(const_magic_val, magic_val + 1) == true);

  CHECK(op(Make<CustomEqT>(magic_val), magic_val) == false);
  CHECK(op(Make<CustomEqT>(magic_val), magic_val + 1) == true);
}
