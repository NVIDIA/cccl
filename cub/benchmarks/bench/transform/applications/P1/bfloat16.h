// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include <nvbench/type_strings.cuh>

// ============================================================================
// BFloat16 type — replicated from c10::BFloat16
// (torch/headeronly/util/BFloat16.h)
// ============================================================================

namespace bf16_detail
{
inline __host__ __device__ float f32_from_bits(uint16_t src)
{
  float res    = 0;
  uint32_t tmp = src;
  tmp <<= 16;
  std::memcpy(&res, &tmp, sizeof(tmp));
  return res;
}

inline __host__ __device__ uint16_t round_to_nearest_even(float src)
{
  if (std::isnan(src))
  {
    return UINT16_C(0x7FC0);
  }
  else
  {
    uint32_t U32;
    std::memcpy(&U32, &src, sizeof(U32));
    uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
  }
}
} // namespace bf16_detail

struct alignas(2) BFloat16
{
  uint16_t x;

  BFloat16() = default;

  struct from_bits_t
  {};
  static constexpr __host__ __device__ from_bits_t from_bits()
  {
    return from_bits_t();
  }

  constexpr __host__ __device__ BFloat16(unsigned short bits, from_bits_t)
      : x(bits)
  {}

  /* implicit */ inline __host__ __device__ BFloat16(float value);
  inline __host__ __device__ operator float() const;

  inline __host__ __device__ BFloat16(const __nv_bfloat16& value);
  explicit inline __host__ __device__ operator __nv_bfloat16() const;
};

inline __host__ __device__ BFloat16::BFloat16(float value)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
                    ({
                      __nv_bfloat16 tmp = __float2bfloat16(value);
                      x                 = *reinterpret_cast<const unsigned short*>(&tmp);
                    }),
                    ({ x = bf16_detail::round_to_nearest_even(value); }));
}

inline __host__ __device__ BFloat16::operator float() const
{
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
}

inline __host__ __device__ BFloat16::BFloat16(const __nv_bfloat16& value)
{
  x = *reinterpret_cast<const unsigned short*>(&value);
}
inline __host__ __device__ BFloat16::operator __nv_bfloat16() const
{
  return *reinterpret_cast<const __nv_bfloat16*>(&x);
}

// Arithmetic — BFloat16 x BFloat16 → BFloat16
inline __host__ __device__ BFloat16 operator+(const BFloat16& a, const BFloat16& b)
{
  return static_cast<float>(a) + static_cast<float>(b);
}
inline __host__ __device__ BFloat16 operator-(const BFloat16& a, const BFloat16& b)
{
  return static_cast<float>(a) - static_cast<float>(b);
}
inline __host__ __device__ BFloat16 operator*(const BFloat16& a, const BFloat16& b)
{
  return static_cast<float>(a) * static_cast<float>(b);
}
inline __host__ __device__ BFloat16 operator/(const BFloat16& a, const BFloat16& b)
{
  return static_cast<float>(a) / static_cast<float>(b);
}
inline __host__ __device__ BFloat16 operator-(const BFloat16& a)
{
  return -static_cast<float>(a);
}

// Compound assignment — BFloat16
inline __host__ __device__ BFloat16& operator+=(BFloat16& a, const BFloat16& b)
{
  a = a + b;
  return a;
}
inline __host__ __device__ BFloat16& operator-=(BFloat16& a, const BFloat16& b)
{
  a = a - b;
  return a;
}
inline __host__ __device__ BFloat16& operator*=(BFloat16& a, const BFloat16& b)
{
  a = a * b;
  return a;
}
inline __host__ __device__ BFloat16& operator/=(BFloat16& a, const BFloat16& b)
{
  a = a / b;
  return a;
}

// Arithmetic — BFloat16 x float → float
inline __host__ __device__ float operator+(BFloat16 a, float b)
{
  return static_cast<float>(a) + b;
}
inline __host__ __device__ float operator-(BFloat16 a, float b)
{
  return static_cast<float>(a) - b;
}
inline __host__ __device__ float operator*(BFloat16 a, float b)
{
  return static_cast<float>(a) * b;
}
inline __host__ __device__ float operator/(BFloat16 a, float b)
{
  return static_cast<float>(a) / b;
}
inline __host__ __device__ float operator+(float a, BFloat16 b)
{
  return a + static_cast<float>(b);
}
inline __host__ __device__ float operator-(float a, BFloat16 b)
{
  return a - static_cast<float>(b);
}
inline __host__ __device__ float operator*(float a, BFloat16 b)
{
  return a * static_cast<float>(b);
}
inline __host__ __device__ float operator/(float a, BFloat16 b)
{
  return a / static_cast<float>(b);
}

// Compound assignment — float x BFloat16 → float
inline __host__ __device__ float& operator+=(float& a, const BFloat16& b)
{
  return a += static_cast<float>(b);
}
inline __host__ __device__ float& operator-=(float& a, const BFloat16& b)
{
  return a -= static_cast<float>(b);
}
inline __host__ __device__ float& operator*=(float& a, const BFloat16& b)
{
  return a *= static_cast<float>(b);
}
inline __host__ __device__ float& operator/=(float& a, const BFloat16& b)
{
  return a /= static_cast<float>(b);
}

// Arithmetic — BFloat16 x int → BFloat16
inline __host__ __device__ BFloat16 operator+(BFloat16 a, int b)
{
  return a + static_cast<BFloat16>(static_cast<float>(b));
}
inline __host__ __device__ BFloat16 operator-(BFloat16 a, int b)
{
  return a - static_cast<BFloat16>(static_cast<float>(b));
}
inline __host__ __device__ BFloat16 operator*(BFloat16 a, int b)
{
  return a * static_cast<BFloat16>(static_cast<float>(b));
}
inline __host__ __device__ BFloat16 operator/(BFloat16 a, int b)
{
  return a / static_cast<BFloat16>(static_cast<float>(b));
}
inline __host__ __device__ BFloat16 operator+(int a, BFloat16 b)
{
  return static_cast<BFloat16>(static_cast<float>(a)) + b;
}
inline __host__ __device__ BFloat16 operator-(int a, BFloat16 b)
{
  return static_cast<BFloat16>(static_cast<float>(a)) - b;
}
inline __host__ __device__ BFloat16 operator*(int a, BFloat16 b)
{
  return static_cast<BFloat16>(static_cast<float>(a)) * b;
}
inline __host__ __device__ BFloat16 operator/(int a, BFloat16 b)
{
  return static_cast<BFloat16>(static_cast<float>(a)) / b;
}

// Comparison — for std::min/std::max
inline __host__ __device__ bool operator>(BFloat16& lhs, BFloat16& rhs)
{
  return float(lhs) > float(rhs);
}
inline __host__ __device__ bool operator<(BFloat16& lhs, BFloat16& rhs)
{
  return float(lhs) < float(rhs);
}

// NVBench type registration
NVBENCH_DECLARE_TYPE_STRINGS(BFloat16, "bf16", "BFloat16");
