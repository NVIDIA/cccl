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

#include <c2h/test_util_vec.h>

/***********************************************************************************************************************
 * CUB operator to identity
 **********************************************************************************************************************/

template <typename T, typename Operator>
inline const T operator_identity_v;

template <typename T>
inline const T operator_identity_v<T, cuda::std::plus<>> = T{};

template <typename T>
inline const T operator_identity_v<T, cuda::std::plus<T>> = T{};

template <typename T>
inline const T operator_identity_v<T, cuda::std::multiplies<>> = T{1};

template <typename T>
inline const T operator_identity_v<T, cuda::std::multiplies<T>> = T{1};

template <typename T>
inline const T operator_identity_v<T, cuda::std::bit_and<>> = static_cast<T>(~T{0});

template <typename T>
inline const T operator_identity_v<T, cuda::std::bit_and<T>> = static_cast<T>(~T{0});

template <typename T>
inline const T operator_identity_v<T, cuda::std::bit_or<>> = T{0};

template <typename T>
inline const T operator_identity_v<T, cuda::std::bit_or<T>> = T{0};

template <typename T>
inline const T operator_identity_v<T, cuda::std::bit_xor<>> = T{0};

template <typename T>
inline const T operator_identity_v<T, cuda::std::bit_xor<T>> = T{0};

template <>
inline const bool operator_identity_v<bool, cuda::std::logical_and<>> = true;

template <>
inline const bool operator_identity_v<bool, cuda::std::logical_and<bool>> = true;

template <>
inline const bool operator_identity_v<bool, cuda::std::logical_or<>> = false;

template <>
inline const bool operator_identity_v<bool, cuda::std::logical_or<bool>> = false;

template <typename T>
inline const T operator_identity_v<T, cuda::maximum<>> = cuda::std::numeric_limits<T>::lowest();

template <typename T>
inline const T operator_identity_v<T, cuda::maximum<T>> = cuda::std::numeric_limits<T>::lowest();

template <typename T>
inline const T operator_identity_v<T, cuda::minimum<>> = cuda::std::numeric_limits<T>::max();

template <typename T>
inline const T operator_identity_v<T, cuda::minimum<T>> = cuda::std::numeric_limits<T>::max();

struct custom_plus : cuda::std::plus<>
{};
