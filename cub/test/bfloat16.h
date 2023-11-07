/******************************************************************************
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

/**
 * \file
 * Utilities for interacting with the opaque CUDA __nv_bfloat16 type
 */

#include <cub/util_type.cuh>

#include <cuda_bf16.h>

#include <cuda/std/type_traits>

#include <cstdint>
#include <iosfwd>

#ifdef __GNUC__
// There's a ton of type-punning going on in this file.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif


/******************************************************************************
 * bfloat16_t
 ******************************************************************************/

/**
 * Host-based fp16 data type compatible and convertible with __nv_bfloat16
 */
struct bfloat16_t
{
    uint16_t __x;

    /// Constructor from __nv_bfloat16
    __host__ __device__ __forceinline__
    bfloat16_t(const __nv_bfloat16 &other)
    {
        __x = reinterpret_cast<const uint16_t&>(other);
    }

    /// Constructor from integer
    __host__ __device__ __forceinline__
    bfloat16_t(int a)
    {
        *this = bfloat16_t(float(a));
    }

    /// Constructor from std::size_t
    __host__ __device__ __forceinline__
    bfloat16_t(std::size_t a)
    {
        *this = bfloat16_t(float(a));
    }

    /// Constructor from unsigned long long int
    template < typename T,
               typename = typename ::cuda::std::enable_if<
                 ::cuda::std::is_same<T, unsigned long long int>::value
                 && (!::cuda::std::is_same<std::size_t, unsigned long long int>::value)>::type>
    __host__ __device__ __forceinline__ bfloat16_t(T a)
    {
      *this = bfloat16_t(float(a));
    }

    /// Default constructor
    bfloat16_t() = default;

    /// Constructor from float
    __host__ __device__ __forceinline__
    bfloat16_t(float a)
    {
        // Refrence:
        // https://github.com/pytorch/pytorch/blob/44cc873fba5e5ffc4d4d4eef3bd370b653ce1ce1/c10/util/BFloat16.h#L51
        uint16_t ir;
        if (a != a) {
            ir = UINT16_C(0x7FFF);
        } else {
            union {
                uint32_t U32;
                float F32;
            };

            F32 = a;
            uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
            ir = static_cast<uint16_t>((U32 + rounding_bias) >> 16);
        }
        this->__x = ir;
    }

    /// Cast to __nv_bfloat16
    __host__ __device__ __forceinline__
    operator __nv_bfloat16() const
    {
        return reinterpret_cast<const __nv_bfloat16&>(__x);
    }

    /// Cast to float
    __host__ __device__ __forceinline__
    operator float() const
    {
        float f = 0;
        uint32_t *p = reinterpret_cast<uint32_t *>(&f);
        *p = uint32_t(__x) << 16;
        return f;
    }


    /// Get raw storage
    __host__ __device__ __forceinline__
    uint16_t raw() const
    {
        return this->__x;
    }

    /// Equality
    __host__ __device__ __forceinline__
    bool operator ==(const bfloat16_t &other) const
    {
        return (this->__x == other.__x);
    }

    /// Inequality
    __host__ __device__ __forceinline__
    bool operator !=(const bfloat16_t &other) const
    {
        return (this->__x != other.__x);
    }

    /// Assignment by sum
    __host__ __device__ __forceinline__
    bfloat16_t& operator +=(const bfloat16_t &rhs)
    {
        *this = bfloat16_t(float(*this) + float(rhs));
        return *this;
    }

    /// Multiply
    __host__ __device__ __forceinline__
    bfloat16_t operator*(const bfloat16_t &other)
    {
        return bfloat16_t(float(*this) * float(other));
    }

    /// Add
    __host__ __device__ __forceinline__
    bfloat16_t operator+(const bfloat16_t &other)
    {
        return bfloat16_t(float(*this) + float(other));
    }

    /// Less-than
    __host__ __device__ __forceinline__
    bool operator<(const bfloat16_t &other) const
    {
        return float(*this) < float(other);
    }

    /// Less-than-equal
    __host__ __device__ __forceinline__
    bool operator<=(const bfloat16_t &other) const
    {
        return float(*this) <= float(other);
    }

    /// Greater-than
    __host__ __device__ __forceinline__
    bool operator>(const bfloat16_t &other) const
    {
        return float(*this) > float(other);
    }

    /// Greater-than-equal
    __host__ __device__ __forceinline__
    bool operator>=(const bfloat16_t &other) const
    {
        return float(*this) >= float(other);
    }

    /// numeric_traits<bfloat16_t>::max
    __host__ __device__ __forceinline__
    static bfloat16_t (max)() {
        uint16_t max_word = 0x7F7F;
        return reinterpret_cast<bfloat16_t&>(max_word);
    }

    /// numeric_traits<bfloat16_t>::lowest
    __host__ __device__ __forceinline__
    static bfloat16_t lowest() {
        uint16_t lowest_word = 0xFF7F;
        return reinterpret_cast<bfloat16_t&>(lowest_word);
    }
};


/******************************************************************************
 * I/O stream overloads
 ******************************************************************************/

/// Insert formatted \p bfloat16_t into the output stream
inline std::ostream& operator<<(std::ostream &out, const bfloat16_t &x)
{
    out << (float)x;
    return out;
}


/// Insert formatted \p __nv_bfloat16 into the output stream
inline std::ostream& operator<<(std::ostream &out, const __nv_bfloat16 &x)
{
    return out << bfloat16_t(x);
}


/******************************************************************************
 * Traits overloads
 ******************************************************************************/

template <>
struct CUB_NS_QUALIFIER::FpLimits<bfloat16_t>
{
    static __host__ __device__ __forceinline__ bfloat16_t Max() { return bfloat16_t::max(); }

    static __host__ __device__ __forceinline__ bfloat16_t Lowest() { return bfloat16_t::lowest(); }
};

template <>
struct CUB_NS_QUALIFIER::NumericTraits<bfloat16_t>
    : CUB_NS_QUALIFIER::
        BaseTraits<FLOATING_POINT, true, false, unsigned short, bfloat16_t>
{};

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
