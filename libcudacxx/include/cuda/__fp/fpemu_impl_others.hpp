//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_OTHERS_H
#define _CUDA___FP_FPEMU_IMPL_OTHERS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/**
 * @file fpemu_impl_others.hpp
 * @brief Implementation of MAD, DOT, and CMUL operations for FPEMU floating point emulation library
 *
 * This header provides the implementation of other operations for the FPEMU library.
 * It includes:
 *   - MAD (Multiply-Add with intermediate rounding) functions for different accuracy and range configurations 
 *   - DOT (dot product) functions
 *   - CMUL (complex multiply) functions
 *   - Special case handling for NaN, inf, zero, etc
 *
 * The implementation is designed to work across both host and device code
 * through appropriate decorators and provide bit-exact results matching hardware
 * floating point units.
 */

 #define __FP64EMU_USE_OPT_MAD_UNPACKED__  1
 #define __FP64EMU_USE_OPT_DOT_UNPACKED__  1
 #define __FP64EMU_USE_OPT_CMUL_UNPACKED__ 1

#include <cuda/__fp/fpemu_common.hpp>
#include <cuda/__fp/fpemu_impl_utils.hpp>
#include <cuda/__fp/fpemu_impl_unpack.hpp>
#include <cuda/__fp/fpemu_impl_mul.hpp>
#include <cuda/__fp/fpemu_impl_add.hpp>
#include <cuda/__fp/fpemu_impl_sub.hpp>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

namespace impl
{
    // MAD unpacked implementation
    template<fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_unpacked_t __nv_internal_fp64emu_mad_unpacked (fpbits64_unpacked_t x, 
                                                          fpbits64_unpacked_t y, 
                                                          fpbits64_unpacked_t z)
    {
        return __nv_internal_fp64emu_dadd_unpacked<meth>(
               __nv_internal_fp64emu_dmul_unpacked<meth>(x, y), z);
    }

    // DOT unpacked implementation
    template<fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_unpacked_t __nv_internal_fp64emu_dot_unpacked (fpbits64_unpacked_t x1, 
                                                          fpbits64_unpacked_t y1, 
                                                          fpbits64_unpacked_t x2,
                                                          fpbits64_unpacked_t y2)
    {
        return __nv_internal_fp64emu_dadd_unpacked<meth>(
               __nv_internal_fp64emu_dmul_unpacked<meth>(x1, x2), 
               __nv_internal_fp64emu_dmul_unpacked<meth>(y1, y2));
    }

    // CMPLX MUL unpacked implementation
    // (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
    template<fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    void __nv_internal_fp64emu_cmul_unpacked (fpbits64_unpacked_t  x_re, 
                                            fpbits64_unpacked_t  x_im, 
                                            fpbits64_unpacked_t  y_re,
                                            fpbits64_unpacked_t  y_im,
                                            fpbits64_unpacked_t& r_re,
                                            fpbits64_unpacked_t& r_im)
    {
        r_re = __nv_internal_fp64emu_dsub_unpacked<meth>(__nv_internal_fp64emu_dmul_unpacked<meth>(x_re, y_re), 
                                                                    __nv_internal_fp64emu_dmul_unpacked<meth>(x_im, y_im));
        r_im = __nv_internal_fp64emu_dadd_unpacked<meth>(__nv_internal_fp64emu_dmul_unpacked<meth>(x_re, y_im), 
                                                                    __nv_internal_fp64emu_dmul_unpacked<meth>(x_im, y_re));
        return;
    }

    // MAD implementation
    template<fpemu::rounding rm   = fpemu::rounding::def, 
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_t __nv_internal_fp64emu_mad (fpbits64_t x, 
                                        fpbits64_t y, 
                                        fpbits64_t z)
    {
        if constexpr (meth == fp64emu_accuracy::mid)
        {
            fpbits64_unpacked_t x_unpacked = __nv_internal_fp64emu_unpack(x);
            fpbits64_unpacked_t y_unpacked = __nv_internal_fp64emu_unpack(y);
            fpbits64_unpacked_t z_unpacked = __nv_internal_fp64emu_unpack(z);

            fpbits64_unpacked_t r_unpacked = __nv_internal_fp64emu_mad_unpacked<meth>(x_unpacked, 
                                                                                                 y_unpacked, 
                                                                                                 z_unpacked);
            return __nv_internal_fp64emu_pack<rm>(r_unpacked);
        }
        else
        {
            return  __nv_internal_fp64emu_dadd<rm, meth>(
                    __nv_internal_fp64emu_dmul<rm, meth>(x, y), z);
        }
    }

    // DOT implementation
    template<fpemu::rounding rm   = fpemu::rounding::def, 
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_t __nv_internal_fp64emu_dot (fpbits64_t x1, 
                                        fpbits64_t y1, 
                                        fpbits64_t x2,
                                        fpbits64_t y2)
    {
        if constexpr (meth == fp64emu_accuracy::mid)
        {
            fpbits64_unpacked_t x1_unpacked = __nv_internal_fp64emu_unpack(x1);
            fpbits64_unpacked_t y1_unpacked = __nv_internal_fp64emu_unpack(y1);
            fpbits64_unpacked_t x2_unpacked = __nv_internal_fp64emu_unpack(x2);
            fpbits64_unpacked_t y2_unpacked = __nv_internal_fp64emu_unpack(y2);

            fpbits64_unpacked_t r_unpacked = __nv_internal_fp64emu_dot_unpacked<meth>(x1_unpacked, 
                                                                                                 y1_unpacked, 
                                                                                                 x2_unpacked,
                                                                                                 y2_unpacked);
            fpbits64_t r = __nv_internal_fp64emu_pack<rm>(r_unpacked);

            return r;
        }
        else
        {
            fpbits64_t r = __nv_internal_fp64emu_dadd<rm, meth>(
                           __nv_internal_fp64emu_dmul<rm, meth>(x1, x2),
                           __nv_internal_fp64emu_dmul<rm, meth>(y1, y2));
            return r;
        }
    }

    // CMUL implementation
    template<fpemu::rounding rm   = fpemu::rounding::def, 
             fp64emu_accuracy   meth = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    void __nv_internal_fp64emu_cmul (fpbits64_t  x_re, 
                                   fpbits64_t  x_im, 
                                   fpbits64_t  y_re, 
                                   fpbits64_t  y_im,
                                   fpbits64_t& r_re,
                                   fpbits64_t& r_im)
    {
        if constexpr (meth == fp64emu_accuracy::mid)
        {
            fpbits64_unpacked_t x_re_unpacked = __nv_internal_fp64emu_unpack(x_re);
            fpbits64_unpacked_t y_re_unpacked = __nv_internal_fp64emu_unpack(y_re);
            fpbits64_unpacked_t x_im_unpacked = __nv_internal_fp64emu_unpack(x_im);
            fpbits64_unpacked_t y_im_unpacked = __nv_internal_fp64emu_unpack(y_im);
            fpbits64_unpacked_t r_re_unpacked;
            fpbits64_unpacked_t r_im_unpacked;

            __nv_internal_fp64emu_cmul_unpacked<meth>(x_re_unpacked,
                                                                 x_im_unpacked,
                                                                 y_re_unpacked, 
                                                                 y_im_unpacked,
                                                                 r_re_unpacked,
                                                                 r_im_unpacked);

            r_re = __nv_internal_fp64emu_pack<rm>(r_re_unpacked);
            r_im = __nv_internal_fp64emu_pack<rm>(r_im_unpacked);

            return;
        }
        else
        {
            fpbits64_t r_re_y_re = __nv_internal_fp64emu_dmul<rm, meth>(x_re, y_re);
            fpbits64_t r_im_y_im = __nv_internal_fp64emu_dmul<rm, meth>(x_im, y_im);
            fpbits64_t r_re_y_im = __nv_internal_fp64emu_dmul<rm, meth>(x_re, y_im);
            fpbits64_t r_im_y_re = __nv_internal_fp64emu_dmul<rm, meth>(x_im, y_re);

            r_re = __nv_internal_fp64emu_dsub<rm, meth>(r_re_y_re, r_im_y_im);
            r_im = __nv_internal_fp64emu_dadd<rm, meth>(r_re_y_im, r_im_y_re);

            return;
        }
    }

    __FPEMU_INTERNAL_DECL__ fpbits64_unpacked_t __nv_internal_fp64emu_neg_unpacked (fpbits64_unpacked_t x) 
    { 
       x.sign = fpemu::__invert_msb(x.sign);
       return x;
    }

    __FPEMU_INTERNAL_DECL__ fpbits64_t __nv_internal_fp64emu_neg (fpbits64_t x) 
    { 
        fpemu::uint32x2_t t = fpemu::bit_cast<fpemu::uint32x2_t>(x);
        t.x[1]              = fpemu::__invert_msb(t.x[1]);
        x                   = fpemu::bit_cast<uint64_t>(t);
        return              x;
    }

} // namespace impl

// ============================================================================
// Builtin declarations/implementations for MAD, DOT, CMUL, NEG operations
// ============================================================================
#if defined(__FPEMU_INLINE__)

// mad (packed)
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mad_rn          (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_mad<fpemu::rounding::rn, fp64emu_accuracy::high>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_mad_rn (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_mad<fpemu::rounding::rn, fp64emu_accuracy::high>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_mad_rn      (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_mad<fpemu::rounding::rn, fp64emu_accuracy::mid>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_mad_rn     (fpbits64_t x, fpbits64_t y, fpbits64_t z) { return impl::__nv_internal_fp64emu_mad<fpemu::rounding::rn, fp64emu_accuracy::low>(x, y, z); }

// dot (packed)
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dot_rn          (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2) { return impl::__nv_internal_fp64emu_dot<fpemu::rounding::rn, fp64emu_accuracy::high>(x1, y1, x2, y2); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_dot_rn (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2) { return impl::__nv_internal_fp64emu_dot<fpemu::rounding::rn, fp64emu_accuracy::high>(x1, y1, x2, y2); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dot_rn      (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2) { return impl::__nv_internal_fp64emu_dot<fpemu::rounding::rn, fp64emu_accuracy::mid>(x1, y1, x2, y2); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dot_rn     (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2) { return impl::__nv_internal_fp64emu_dot<fpemu::rounding::rn, fp64emu_accuracy::low>(x1, y1, x2, y2); }

// cmul (packed)
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_cmul_rn          (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im) { impl::__nv_internal_fp64emu_cmul<fpemu::rounding::rn, fp64emu_accuracy::mid>(x_re, x_im, y_re, y_im, r_re, r_im); }
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_high_cmul_rn (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im) { impl::__nv_internal_fp64emu_cmul<fpemu::rounding::rn, fp64emu_accuracy::high>(x_re, x_im, y_re, y_im, r_re, r_im); }
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_mid_cmul_rn      (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im) { impl::__nv_internal_fp64emu_cmul<fpemu::rounding::rn, fp64emu_accuracy::mid>(x_re, x_im, y_re, y_im, r_re, r_im); }
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_low_cmul_rn     (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im) { impl::__nv_internal_fp64emu_cmul<fpemu::rounding::rn, fp64emu_accuracy::low>(x_re, x_im, y_re, y_im, r_re, r_im); }

// neg (packed)
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_neg (fpbits64_t x) { return impl::__nv_internal_fp64emu_neg(x); }

#if __FPEMU_UNPACKED__ == 1
// mad (unpacked)
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mad          (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z) { return impl::__nv_internal_fp64emu_mad_unpacked<fp64emu_accuracy::mid>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_mad (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z) { return impl::__nv_internal_fp64emu_mad_unpacked<fp64emu_accuracy::high>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_mad      (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z) { return impl::__nv_internal_fp64emu_mad_unpacked<fp64emu_accuracy::mid>(x, y, z); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_mad     (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z) { return impl::__nv_internal_fp64emu_mad_unpacked<fp64emu_accuracy::low>(x, y, z); }

// dot (unpacked)
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dot          (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2) { return impl::__nv_internal_fp64emu_dot_unpacked<fp64emu_accuracy::mid>(x1, y1, x2, y2); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_dot (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2) { return impl::__nv_internal_fp64emu_dot_unpacked<fp64emu_accuracy::high>(x1, y1, x2, y2); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dot      (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2) { return impl::__nv_internal_fp64emu_dot_unpacked<fp64emu_accuracy::mid>(x1, y1, x2, y2); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dot     (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2) { return impl::__nv_internal_fp64emu_dot_unpacked<fp64emu_accuracy::low>(x1, y1, x2, y2); }

// cmul (unpacked)
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_unpacked_cmul          (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im) { impl::__nv_internal_fp64emu_cmul_unpacked<fp64emu_accuracy::mid>(x_re, x_im, y_re, y_im, r_re, r_im); }
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_unpacked_high_cmul (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im) { impl::__nv_internal_fp64emu_cmul_unpacked<fp64emu_accuracy::high>(x_re, x_im, y_re, y_im, r_re, r_im); }
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_unpacked_mid_cmul      (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im) { impl::__nv_internal_fp64emu_cmul_unpacked<fp64emu_accuracy::mid>(x_re, x_im, y_re, y_im, r_re, r_im); }
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_unpacked_low_cmul     (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im) { impl::__nv_internal_fp64emu_cmul_unpacked<fp64emu_accuracy::low>(x_re, x_im, y_re, y_im, r_re, r_im); }

// neg (unpacked)
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_neg (fpbits64_unpacked_t x) { return impl::__nv_internal_fp64emu_neg_unpacked(x); }
#endif // __FPEMU_UNPACKED__ == 1

#else // LTO mode - declarations only

// mad (packed)
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mad_rn          (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_mad_rn (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_mad_rn      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_mad_rn     (fpbits64_t x, fpbits64_t y, fpbits64_t z);

// dot (packed)
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dot_rn          (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_dot_rn (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dot_rn      (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dot_rn     (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2);

// cmul (packed)
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_cmul_rn          (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im);
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_high_cmul_rn (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im);
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_mid_cmul_rn      (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im);
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_low_cmul_rn     (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im);

// neg (packed)
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_neg (fpbits64_t x);

#if __FPEMU_UNPACKED__ == 1
// mad (unpacked)
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mad          (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_mad (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_mad      (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_mad     (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);

// dot (unpacked)
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dot          (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_dot (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dot      (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dot     (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2);

// cmul (unpacked)
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_unpacked_cmul          (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im);
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_unpacked_high_cmul (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im);
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_unpacked_mid_cmul      (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im);
__FPEMU_BUILTIN_DECL__ void __nv_fp64emu_unpacked_low_cmul     (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im);

// neg (unpacked)
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_neg (fpbits64_unpacked_t x);
#endif // __FPEMU_UNPACKED__ == 1

#endif // __FPEMU_INLINE__

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_OTHERS_HPP__

#if defined(__FPEMU_API_CLASSES_DEFINED__) && !defined(__FPEMU_OTHERS_API_MERGED__)
#define __FPEMU_OTHERS_API_MERGED__
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


// ============================================================================
// API (merged from fp64emu_others_api.hpp)
// ============================================================================

    // Unary negation operator - member function implementation
    template<fp64emu_accuracy m>
    __FPEMU_HOST_DEVICE_DECL__ fp64emu_t<m> fp64emu_t<m>::operator-() const
    {
        fp64emu_t temp(*this);
        temp.bits = __nv_fp64emu_neg(temp.bits);
        return temp;
    }


    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> mad (const fp64emu_t<m>& x, const fp64emu_t<m>& y, const fp64emu_t<m>& z) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_high_mad_rn(x.bits, y.bits, z.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_mad_rn(x.bits, y.bits, z.bits)); }
        else                                      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_mad_rn(x.bits, y.bits, z.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> __mad_rn (const fp64emu_t<m>& x, const fp64emu_t<m>& y, const fp64emu_t<m>& z) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_high_mad_rn(x.bits, y.bits, z.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_mad_rn(x.bits, y.bits, z.bits)); }
        else                                      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_mad_rn(x.bits, y.bits, z.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_t<m> dot (const fp64emu_t<m>& x1, const fp64emu_t<m>& y1, const fp64emu_t<m>& x2, const fp64emu_t<m>& y2) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_high_dot_rn(x1.bits, y1.bits, x2.bits, y2.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_low_dot_rn(x1.bits, y1.bits, x2.bits, y2.bits)); }
        else                                      { return fp64emu_t<m>(fpbits64_construct, __nv_fp64emu_mid_dot_rn(x1.bits, y1.bits, x2.bits, y2.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ void cmul (const fp64emu_t<m>& x_re, const fp64emu_t<m>& x_im, const fp64emu_t<m>& y_re, const fp64emu_t<m>& y_im, fp64emu_t<m>& r_re, fp64emu_t<m>& r_im) { 
        if      constexpr (m == fp64emu_accuracy::high) { __nv_fp64emu_high_cmul_rn(x_re.bits, x_im.bits, y_re.bits, y_im.bits, r_re.bits, r_im.bits); }
        else if constexpr (m == fp64emu_accuracy::low)     { __nv_fp64emu_low_cmul_rn(x_re.bits, x_im.bits, y_re.bits, y_im.bits, r_re.bits, r_im.bits); }
        else                                      { __nv_fp64emu_mid_cmul_rn(x_re.bits, x_im.bits, y_re.bits, y_im.bits, r_re.bits, r_im.bits); }
    } 

#if __FPEMU_UNPACKED__ == 1

    // Stream insertion for unpacked type (deferred from class definition)
    template<fp64emu_accuracy m>
    __FPEMU_HOST_DEVICE_DECL__ inline ::std::ostream& operator<<(::std::ostream& os, const fp64emu_unpacked_t<m>& ef)
    {
        double d = __nv_fp64emu_unpacked_to_double(ef.bits);
        os << d;
        return os;
    }

    // Unary negation operator for unpacked - member function implementation
    template<fp64emu_accuracy m>
    __FPEMU_HOST_DEVICE_DECL__ fp64emu_unpacked_t<m> fp64emu_unpacked_t<m>::operator-() const
    {
        fp64emu_unpacked_t temp(*this);
        temp.bits = __nv_fp64emu_unpacked_neg(temp.bits);
        return temp;
    }


    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_unpacked_t<m> mad (const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y, const fp64emu_unpacked_t<m>& z) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_high_mad(x.bits, y.bits, z.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_low_mad(x.bits, y.bits, z.bits)); }
        else                                      { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_mid_mad(x.bits, y.bits, z.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_unpacked_t<m> __mad_rn (const fp64emu_unpacked_t<m>& x, const fp64emu_unpacked_t<m>& y, const fp64emu_unpacked_t<m>& z) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_high_mad(x.bits, y.bits, z.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_low_mad(x.bits, y.bits, z.bits)); }
        else                                      { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_mid_mad(x.bits, y.bits, z.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ fp64emu_unpacked_t<m> dot (const fp64emu_unpacked_t<m>& x1, const fp64emu_unpacked_t<m>& y1, const fp64emu_unpacked_t<m>& x2, const fp64emu_unpacked_t<m>& y2) { 
        if      constexpr (m == fp64emu_accuracy::high) { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_high_dot(x1.bits, y1.bits, x2.bits, y2.bits)); }
        else if constexpr (m == fp64emu_accuracy::low)     { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_low_dot(x1.bits, y1.bits, x2.bits, y2.bits)); }
        else                                      { return fp64emu_unpacked_t<m>(fpbits64_construct, __nv_fp64emu_unpacked_mid_dot(x1.bits, y1.bits, x2.bits, y2.bits)); }
    }
    template<fp64emu_accuracy m>
    __FPEMU_API_DECL__ void cmul (const fp64emu_unpacked_t<m>& x_re, const fp64emu_unpacked_t<m>& x_im, const fp64emu_unpacked_t<m>& y_re, const fp64emu_unpacked_t<m>& y_im, fp64emu_unpacked_t<m>& r_re, fp64emu_unpacked_t<m>& r_im) { 
        if      constexpr (m == fp64emu_accuracy::high) { __nv_fp64emu_unpacked_high_cmul(x_re.bits, x_im.bits, y_re.bits, y_im.bits, r_re.bits, r_im.bits); }
        else if constexpr (m == fp64emu_accuracy::low)     { __nv_fp64emu_unpacked_low_cmul(x_re.bits, x_im.bits, y_re.bits, y_im.bits, r_re.bits, r_im.bits); }
        else                                      { __nv_fp64emu_unpacked_mid_cmul(x_re.bits, x_im.bits, y_re.bits, y_im.bits, r_re.bits, r_im.bits); }
    }


#endif // __FPEMU_UNPACKED__ == 1

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_OTHERS_API_MERGED__
