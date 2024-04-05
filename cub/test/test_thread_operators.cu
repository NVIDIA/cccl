/*******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "test_util.h"

#include <cub/thread/thread_operators.cuh>

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

  __host__ __device__ operator int() const { return m_val; }
};

#define CUSTOM_TYPE_FACTORY(NAME, RT, OP, CONVERTABLE)                         \
  class Custom##NAME##T : public BaseT<CONVERTABLE>                            \
  {                                                                            \
    explicit Custom##NAME##T(int val)                                          \
        : BaseT<CONVERTABLE>(val)                                              \
    {}                                                                         \
                                                                               \
    friend Custom##NAME##T Make<Custom##NAME##T>(int);                         \
                                                                               \
  public:                                                                      \
    __host__ __device__ RT operator OP(int val) const                          \
    {                                                                          \
      return m_val OP val;                                                     \
    }                                                                          \
  }

//                  NAME  RT    OP  CONVERTABLE
CUSTOM_TYPE_FACTORY(Eq,   bool, ==, false);
CUSTOM_TYPE_FACTORY(Ineq, bool, !=, false);
CUSTOM_TYPE_FACTORY(Sum,  int,  +,  false);
CUSTOM_TYPE_FACTORY(Diff, int,  -,  false);
CUSTOM_TYPE_FACTORY(Div,  int,  /,  false);
CUSTOM_TYPE_FACTORY(Gt,   bool, >,  true);
CUSTOM_TYPE_FACTORY(Lt,   bool, <,  true);

void TestEquality()
{
  cub::Equality op{}; 

  constexpr int const_magic_val = 42;
  int magic_val = const_magic_val;

  AssertEquals(op(const_magic_val, const_magic_val), true);
  AssertEquals(op(const_magic_val, magic_val), true);
  AssertEquals(op(const_magic_val, magic_val + 1), false);

  AssertEquals(op(Make<CustomEqT>(magic_val), magic_val), true);
  AssertEquals(op(Make<CustomEqT>(magic_val), magic_val + 1), false);
}

void TestInequality()
{
  cub::Inequality op{}; 

  constexpr int const_magic_val = 42;
  int magic_val = const_magic_val;

  AssertEquals(op(const_magic_val, const_magic_val), false);
  AssertEquals(op(const_magic_val, magic_val), false);
  AssertEquals(op(const_magic_val, magic_val + 1), true);

  AssertEquals(op(Make<CustomIneqT>(magic_val), magic_val), false);
  AssertEquals(op(Make<CustomIneqT>(magic_val), magic_val + 1), true);
}

void TestInequalityWrapper()
{
  cub::Equality wrapped_op{}; 
  cub::InequalityWrapper<cub::Equality> op{wrapped_op};

  constexpr int const_magic_val = 42;
  int magic_val = const_magic_val;

  AssertEquals(op(const_magic_val, const_magic_val), false);
  AssertEquals(op(const_magic_val, magic_val), false);
  AssertEquals(op(const_magic_val, magic_val + 1), true);

  AssertEquals(op(Make<CustomEqT>(magic_val), magic_val), false);
  AssertEquals(op(Make<CustomEqT>(magic_val), magic_val + 1), true);
}

#define CUSTOM_SYNC_T(NAME, RT, OP)                                            \
  struct Custom ## NAME ## Sink                                                \
  {                                                                            \
    template <class T>                                                         \
    __host__ __device__ RT operator OP (T &&) const                            \
    {                                                                          \
      return RT{};                                                             \
    }                                                                          \
  }

CUSTOM_SYNC_T(SumInt, int, +);
CUSTOM_SYNC_T(SumCustomInt, CustomSumIntSink, +);

CUSTOM_SYNC_T(DiffInt, int, -);
CUSTOM_SYNC_T(DiffCustomInt, CustomDiffIntSink, -);

CUSTOM_SYNC_T(DivInt, int, /);
CUSTOM_SYNC_T(DivCustomInt, CustomDivIntSink, /);

template <class ExpectedT, class ActualT>
void StaticSame()
{
  static_assert(std::is_same<ExpectedT, ActualT>::value, "shall match");
}

void TestSum()
{
  cub::Sum op{};

  constexpr int const_magic_val = 40;
  int magic_val = const_magic_val;

  AssertEquals(op(const_magic_val, 2), 42);
  AssertEquals(op(magic_val, 2), 42);
  AssertEquals(op(Make<CustomSumT>(magic_val), 2), 42);

  StaticSame<decltype(op(42, 42)), int>();
  StaticSame<decltype(op(1, 1.0)), double>();
  StaticSame<decltype(op(CustomSumIntSink{}, 1.0)), int>();
  StaticSame<decltype(op(CustomSumCustomIntSink{}, 1.0)), CustomSumIntSink>();
}

void TestDifference()
{
  cub::Difference op{};

  constexpr int const_magic_val = 44;
  int magic_val = const_magic_val;

  AssertEquals(op(const_magic_val, 2), 42);
  AssertEquals(op(magic_val, 2), 42);

  AssertEquals(op(Make<CustomDiffT>(magic_val), 2), 42);

  StaticSame<decltype(op(42, 42)), int>();
  StaticSame<decltype(op(1, 1.0)), double>();
  StaticSame<decltype(op(CustomDiffIntSink{}, 1.0)), int>();
  StaticSame<decltype(op(CustomDiffCustomIntSink{}, 1.0)), CustomDiffIntSink>();
}

void TestDivision()
{
  cub::Division op{};

  constexpr int const_magic_val = 44;
  int magic_val = const_magic_val;

  AssertEquals(op(const_magic_val, 2), 22);
  AssertEquals(op(magic_val, 2), 22);

  AssertEquals(op(Make<CustomDivT>(magic_val), 2), 22);

  StaticSame<decltype(op(42, 42)), int>();
  StaticSame<decltype(op(1, 1.0)), double>();
  StaticSame<decltype(op(CustomDivIntSink{}, 1.0)), int>();
  StaticSame<decltype(op(CustomDivCustomIntSink{}, 1.0)), CustomDivIntSink>();
}

void TestMax()
{
  cub::Max op{};

  constexpr int const_magic_val = 42;
  int magic_val = const_magic_val;

  AssertEquals(op(const_magic_val, 2), 42);
  AssertEquals(op(magic_val, 2), 42);

  AssertEquals(op(2, Make<CustomGtT>(magic_val)), 42);

  StaticSame<decltype(op(42, 42)), int>();
  StaticSame<decltype(op(1, 1.0)), double>();
  StaticSame<decltype(op(1, Make<CustomGtT>(magic_val))), int>();
}

void TestMin()
{
  cub::Min op{};

  constexpr int const_magic_val = 42;
  int magic_val = const_magic_val;

  AssertEquals(op(const_magic_val, 2), 2);
  AssertEquals(op(magic_val, 2), 2);

  AssertEquals(op(2, Make<CustomLtT>(magic_val)), 2);

  StaticSame<decltype(op(42, 42)), int>();
  StaticSame<decltype(op(1, 1.0)), double>();
  StaticSame<decltype(op(1, Make<CustomLtT>(magic_val))), int>();
}

int main()
{
  TestEquality();
  TestInequality();
  TestInequalityWrapper();
  TestSum();
  TestDifference();
  TestDivision();
  TestMax();
  TestMin();

  return 0;
}
