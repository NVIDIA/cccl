//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>


//  Assumption: minValue < maxValue
//  Assumption: minValue <= rhs <= maxValue
//  Assumption: minValue <= lhs <= maxValue
//  Assumption: minValue >= 0
template <typename T, T minValue, T maxValue>
__host__ __device__
T euclidian_addition(T rhs, T lhs)
{
    const T modulus = maxValue - minValue + 1;
    T ret = rhs + lhs;
    if (ret > maxValue)
        ret -= modulus;
    return ret;
}

template <typename T, T minValue, T maxValue, bool sign = cuda::std::is_signed<T>::value, T zero = 0>
struct signed_euclidean_subtraction {
    static constexpr T modulus = maxValue - minValue + 1;
    __host__ __device__ T operator()(T lhs, T rhs) {
        T ret = lhs - rhs;
        if (ret < minValue) {
            ret += modulus;
        }
        if (ret > maxValue) {
            ret += modulus;
        }
        return ret;
    }
};

template <typename T, T maxValue, T zero>
struct signed_euclidean_subtraction<T, zero, maxValue, false, zero> {
    static constexpr T modulus = maxValue + 1;
    __host__ __device__ T operator()(T lhs, T rhs) {
        T ret = lhs - rhs;
        if (ret > maxValue) {
            ret += modulus;
        }
        return ret;
    }
};

//  Assumption: minValue < maxValue
//  Assumption: minValue <= rhs <= maxValue
//  Assumption: minValue <= lhs <= maxValue
//  Assumption: minValue >= 0
template <typename T, T minValue, T maxValue>
__host__ __device__
T euclidian_subtraction(T lhs, T rhs)
{
    signed_euclidean_subtraction<T, minValue, maxValue> op;

    return op(lhs, rhs);
}
