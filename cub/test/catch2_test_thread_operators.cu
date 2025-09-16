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
