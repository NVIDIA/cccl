// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_FUNCTIONAL_SHUFFLE_SAFETY_H
#define _CUDA_FUNCTIONAL_SHUFFLE_SAFETY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/bit>
#include<cuda/std/array>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/__ptx/instructions/generated/get_sreg.h>
#include <cuda/__cmath/ceil_div.h>


#define _CCCL_HAS_CUDA_COMPILER 1 //fix for now -- to be deleted later

#if _CCCL_HAS_CUDA_COMPILER
_LIBCUDACXX_BEGIN_NAMESPACE_CUDA
    template <typename T>
    constexpr bool __can_warp_shuffle_v = (_CUDA_VSTD::is_trivially_copyable_v<T>::value && sizeof(T) >= sizeof(CUDA_VSTD::uint32_t)) || 
        _CUDA_VSTD::__is_extended_floating_point_v<T>;

    //Input validation for shuffle operations
    void _CCCL_DEVICE validate_width_mask(_CUDA_VSTD::int32_t __w, _CUDA_VSTD::uint32_t __m)
    {
        _CCCL_ASSERT((__w >= 0), "Width must be greater than or equal to zero"); 
        _CCCL_ASSERT((__w <= warpSize), "Width must not exceed warp size"); 
        _CCCL_ASSERT((__m & __activemask()) == __m, "Mask must be a subset of the active mask"); 
        _CCCL_ASSERT(_CUDA_VSTD::has_single_bit(__w), "Width must be a power of two");
    }

    template<typename Tw, _CUDA_VSTD::int32_t __num_Elements>
    _CCCL_DEVICE _CUDA_VSTD::array<_CUDA_VSTD::uint32_t, __num_Elements> __to_32bitBuffer(Tw& __v)
    {
        _CUDA_VSTD::array<_CUDA_VSTD::uint32_t, __num_Elements> __uint_buffer;
        _CUDA_VSTD::memcpy(__uint_buffer.data(), &__v, sizeof(Tw));
        return __uint_buffer;
    }

    template<typename Tw>
    _CCCL_DEVICE Tw __from_32bitBuffer(_CUDA_VSTD::uint32_t* __inA)
    {
        Tw __v;
        // var = _CUDA_VSTD::bit_cast<T>(inArray);
        _CUDA_VSTD::memcpy(&__v, __inA, sizeof(Tw));
        return __v;
    }

    template <typename T>
    _CCCL_DEVICE T shfl(T __var, _CUDA_VSTD::int32_t __srcLane, _CUDA_VSTD::uint32_t __mask = 0xFFFFFFFF, _CUDA_VSTD::int32_t __width = warpSize)
    {
        _CCCL_ASSERT(__can_warp_shuffle_v<T>, "T must be a supported type for warp shuffle operations");
        validate_width_mask(__width, __mask);
        _CCCL_ASSERT((__srcLane >= 0 && __srcLane < __width), "srcLane must be in the range [0, width)"); 
        _CCCL_ASSERT((__mask >> __srcLane) & 1, "srcLane must be part of the mask");

        _CUDA_VSTD::int32_t __num_Elements = _CUDA_VSTD::bit_cast<_CUDA_VSTD::int32_t>
            (cuda::ceil_div(sizeof(T), sizeof(_CUDA_VSTD::uint32_t)));
        _CUDA_VSTD::array<_CUDA_VSTD::uint32_t, __num_Elements> __uint_buffer = {};
        __uint_buffer = __to_32bitBuffer<T, __num_Elements>(__var);
        
        #pragma unroll
        for(int __i = 0; __i < __num_Elements; __i++)
        {
            __uint_buffer[__i] = __shfl_sync(__mask, __uint_buffer[__i], __srcLane, __width);
        }    
        return __from_32bitBuffer<T>(__uint_buffer.data());
    }

    template <typename T>
    _CCCL_DEVICE T shfl_up(T __var, _CUDA_VSTD::int32_t __delta, _CUDA_VSTD::uint32_t __mask = 0xFFFFFFFF, _CUDA_VSTD::int32_t __width = warpSize)
    {
        _CCCL_ASSERT(__can_warp_shuffle_v<T>, "T must be a supported type for warp shuffle operations"); 
        validate_width_mask(__width, __mask); 
        _CCCL_ASSERT((__delta > 0 && __delta < __width), "delta must be in the range (0, width)");

        auto __lid = cuda::ptx::get_sreg_laneid();
        auto __tl = (__lid - __delta) > 0 ? (__lid - __delta) : 0;
        _CCCL_ASSERT((__mask >> __tl) & 1, "TargetLane must be part of the mask");

        _CUDA_VSTD::int32_t __num_Elements = _CUDA_VSTD::bit_cast<_CUDA_VSTD::int32_t>
            (cuda::ceil_div(sizeof(T), sizeof(_CUDA_VSTD::uint32_t)));
        _CUDA_VSTD::array<_CUDA_VSTD::uint32_t, __num_Elements> __uint_buffer = {};
        __uint_buffer = __to_32bitBuffer<T, __num_Elements>(__v);

        #pragma unroll
        for(int __i = 0; __i < __num_Elements; __i++)
        {
            __uint_buffer[__i] = __shfl_up_sync(__m, __uint_buffer[__i], __d, __w);
        }
        return __from_32bitBuffer<T>(__uint_buffer.data());
    }

    template <typename T>
    _CCCL_DEVICE T shfl_down(T __var, _CUDA_VSTD::int32_t __delta, _CUDA_VSTD::uint32_t __mask = 0xFFFFFFFF, _CUDA_VSTD::int32_t __width = warpSize)
    {
        _CCCL_ASSERT(__can_warp_shuffle_v<T>, "T must be a supported type for warp shuffle operations");
        validate_width_mask(__width, __mask); 
        _CCCL_ASSERT((__delta > 0 && __delta < __width), "delta must be in the range (0, width)"); 
    
        auto __lid = cuda::ptx::get_sreg_laneid();
        auto __tl = (__lid + __delta) < __width ? (__lid + __delta) : __width;
        _CCCL_ASSERT((__mask >> __tl) & 1, "TargetLane must be part of the mask");

        _CUDA_VSTD::int32_t __num_Elements = _CUDA_VSTD::bit_cast<_CUDA_VSTD::int32_t>
            (cuda::ceil_div(sizeof(T), sizeof(_CUDA_VSTD::uint32_t)));
        _CUDA_VSTD::array<_CUDA_VSTD::uint32_t, __num_Elements> __uint_buffer = {};
        __uint_buffer = __to_32bitBuffer<T, __num_Elements>(__var);

        #pragma unroll
        for(int __i = 0; __i < __num_Elements; __i++)
        {
            __uint_buffer[__i] = __shfl_down_sync(__mask, __uint_buffer[__i], __delta, __width);
        }
        return __from_32bitBuffer<T>(__uint_buffer.data());
    }

    template <typename T>
    _CCCL_DEVICE T shfl_xor(T __var, _CUDA_VSTD::int32_t __lane_mask, _CUDA_VSTD::uint32_t __mask = 0xFFFFFFFF, _CUDA_VSTD::int32_t __width = warpSize)
    {
        _CCCL_ASSERT(__can_warp_shuffle_v<T>, "T must be a supported type for warp shuffle operations"); 
        validate_width_mask(__width, __mask);
        
        auto __lid = cuda::ptx::get_sreg_laneid();
        auto __clamped_val = _CUDA_VSTD::clamp(__lid ^ __lm, 0, __w);
        _CCCL_ASSERT((__mask >> __clamped_val) & 1, "Clamped Value must be part of the mask");

        _CUDA_VSTD::int32_t __num_Elements = _CUDA_VSTD::bit_cast<_CUDA_VSTD::int32_t>
            (cuda::ceil_div(sizeof(T), sizeof(_CUDA_VSTD::uint32_t)));
        _CUDA_VSTD::array<_CUDA_VSTD::uint32_t, __num_Elements> __uint_buffer = {};
        __uint_buffer = __to_32bitBuffer<T, __num_Elements>(__var);

        #pragma unroll
        for(int __i = 0; __i < __num_Elements; __i++)
        {
            __uint_buffer[__i] = __shfl_xor_sync(__mask, __uint_buffer[__i], __lane_mask, __width);
        }
        return __from_32bitBuffer<T>(__uint_buffer.data());
    }

_LIBCUDACXX_END_NAMESPACE_CUDA
#endif // _CCCL_HAS_CUDA_COMPILER

#endif // _CUDA_FUNCTIONAL_SHUFFLE_SAFETY_H
