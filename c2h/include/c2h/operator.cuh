/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/limits>

/***********************************************************************************************************************
 * CUB operator to identity
 **********************************************************************************************************************/

template <typename T, typename Operator>
inline constexpr T operator_identity_v;

template <typename T>
inline constexpr T operator_identity_v<T, cuda::std::plus<>> = T{};

template <typename T>
inline constexpr T operator_identity_v<T, cuda::std::plus<T>> = T{};

template <typename T>
inline constexpr T operator_identity_v<T, cuda::std::multiplies<>> = T{1};

template <typename T>
inline constexpr T operator_identity_v<T, cuda::std::multiplies<T>> = T{1};

template <typename T>
inline constexpr T operator_identity_v<T, cuda::std::bit_and<>> = T{~T{0}};

template <typename T>
inline constexpr T operator_identity_v<T, cuda::std::bit_and<T>> = T{~T{0}};

template <typename T>
inline constexpr T operator_identity_v<T, cuda::std::bit_or<>> = T{0};

template <typename T>
inline constexpr T operator_identity_v<T, cuda::std::bit_or<T>> = T{0};

template <typename T>
inline constexpr T operator_identity_v<T, cuda::std::bit_xor<>> = T{0};

template <typename T>
inline constexpr T operator_identity_v<T, cuda::std::bit_xor<T>> = T{0};

template <typename T>
inline constexpr T operator_identity_v<T, cuda::maximum<>> = ::std::numeric_limits<T>::min();

template <typename T>
inline constexpr T operator_identity_v<T, cuda::maximum<T>> = ::std::numeric_limits<T>::min();

template <typename T>
inline constexpr T operator_identity_v<T, cuda::minimum<>> = ::std::numeric_limits<T>::max();

template <typename T>
inline constexpr T operator_identity_v<T, cuda::minimum<T>> = ::std::numeric_limits<T>::max();

struct custom_plus : cuda::std::plus<>
{};
