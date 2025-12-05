//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// This header contains a preview of a portability system that enables
// CUDA C++ development with NVC++, NVCC, and supported host compilers.
// These interfaces are not guaranteed to be stable.

#ifndef __NV_PREPROCESSOR
#define __NV_PREPROCESSOR

#if defined(__GNUC__)
#  pragma GCC system_header
#endif

// For all compilers and dialects this header defines:
//  _NV_HAS_INCLUDE
//  _NV_HAS_INCLUDE_NEXT
//  _NV_EVAL
//  _NV_IF
//  _NV_CONCAT_EVAL
// For C++11 and up it defines:
//  _NV_STRIP_PAREN
//  _NV_DISPATCH_N_ARY
//  _NV_FIRST_ARG
//  _NV_REMOVE_PAREN

#if defined(__has_include)
#  define _NV_HAS_INCLUDE(...) __has_include(__VA_ARGS__)
#else
#  define _NV_HAS_INCLUDE(...) 0
#endif

#if defined(__has_include_next)
#  define _NV_HAS_INCLUDE_NEXT(...) __has_include_next(__VA_ARGS__)
#else
#  define _NV_HAS_INCLUDE_NEXT(...) 0
#endif

#define _NV_EVAL1(...) __VA_ARGS__
#define _NV_EVAL(...)  _NV_EVAL1(__VA_ARGS__)

#define _NV_CONCAT_EVAL1(l, r) _NV_EVAL(l##r)
#define _NV_CONCAT_EVAL(l, r)  _NV_CONCAT_EVAL1(l, r)

#define _NV_IF_0(t, f) f
#define _NV_IF_1(t, f) t

#define _NV_IF_BIT(b)           _NV_EVAL(_NV_IF_##b)
#define _NV_IF__EVAL(fn, t, f)  _NV_EVAL(fn(t, f))
#define _NV_IF_EVAL(cond, t, f) _NV_IF__EVAL(_NV_IF_BIT(cond), t, f)

#define _NV_IF1(cond, t, f) _NV_IF_EVAL(cond, t, f)
#define _NV_IF(cond, t, f)  _NV_IF1(_NV_EVAL(cond), _NV_EVAL(t), _NV_EVAL(f))

// The below mechanisms were derived from: https://gustedt.wordpress.com/2010/06/08/detect-empty-macro-arguments/

#define _NV_ARG32(...) _NV_EVAL(_NV_ARG32_0(__VA_ARGS__))
#define _NV_ARG32_0( \
  _0,                \
  _1,                \
  _2,                \
  _3,                \
  _4,                \
  _5,                \
  _6,                \
  _7,                \
  _8,                \
  _9,                \
  _10,               \
  _11,               \
  _12,               \
  _13,               \
  _14,               \
  _15,               \
  _16,               \
  _17,               \
  _18,               \
  _19,               \
  _20,               \
  _21,               \
  _22,               \
  _23,               \
  _24,               \
  _25,               \
  _26,               \
  _27,               \
  _28,               \
  _29,               \
  _30,               \
  _31,               \
  ...)               \
  _31

#define _NV_HAS_COMMA(...) \
  _NV_ARG32(__VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0)

#define _NV_TRIGGER_PARENTHESIS_(...) ,

/*
This tests a variety of conditions for determining what the incoming statement is.
1. test if there is just one argument
2. test if _TRIGGER_PARENTHESIS_ together with the argument adds a comma
3. test if the argument together with a parenthesis adds a comma
4. test if placing it between _TRIGGER_PARENTHESIS_ and the parenthesis adds a comma
*/
#define _NV_ISEMPTY(...)                                                      \
  _NV_ISEMPTY0(_NV_EVAL(_NV_HAS_COMMA(__VA_ARGS__)),                          \
               _NV_EVAL(_NV_HAS_COMMA(_NV_TRIGGER_PARENTHESIS_ __VA_ARGS__)), \
               _NV_EVAL(_NV_HAS_COMMA(__VA_ARGS__(/*empty*/))),               \
               _NV_EVAL(_NV_HAS_COMMA(_NV_TRIGGER_PARENTHESIS_ __VA_ARGS__(/*empty*/))))

#define _NV_PASTE5(_0, _1, _2, _3, _4) _0##_1##_2##_3##_4
#define _NV_ISEMPTY0(_0, _1, _2, _3)   _NV_HAS_COMMA(_NV_PASTE5(_NV_IS_EMPTY_CASE_, _0, _1, _2, _3))
#define _NV_IS_EMPTY_CASE_0001         ,

#define _NV_REMOVE_PAREN(...) _NV_REMOVE_PAREN1(__VA_ARGS__)
#define _NV_REMOVE_PAREN1(...) \
  _NV_STRIP_PAREN(_NV_IF(_NV_TEST_PAREN(__VA_ARGS__), (_NV_STRIP_PAREN(__VA_ARGS__)), (__VA_ARGS__)))

#define _NV_STRIP_PAREN2(...) __VA_ARGS__
#define _NV_STRIP_PAREN1(...) _NV_STRIP_PAREN2 __VA_ARGS__
#define _NV_STRIP_PAREN(...)  _NV_STRIP_PAREN1(__VA_ARGS__)

#define _NV_TEST_PAREN(...)  _NV_TEST_PAREN1(__VA_ARGS__)
#define _NV_TEST_PAREN1(...) _NV_TEST_PAREN2(_NV_TEST_PAREN_DUMMY __VA_ARGS__)
#define _NV_TEST_PAREN2(...) _NV_TEST_PAREN3(_NV_CONCAT_EVAL(_, __VA_ARGS__))
#define _NV_TEST_PAREN3(...) _NV_EVAL(_NV_FIRST_ARG(__VA_ARGS__))

#define __NV_PAREN_YES 1
#define __NV_PAREN_NO  0

#define _NV_TEST_PAREN_DUMMY(...) _NV_PAREN_YES
#define __NV_TEST_PAREN_DUMMY     __NV_PAREN_NO,

#define _NV_FIRST_ARG1(x, ...) x
#define _NV_FIRST_ARG(x, ...)  _NV_FIRST_ARG1(x)

#define _NV_REMOVE_FIRST_ARGS1(...)   __VA_ARGS__
#define _NV_REMOVE_FIRST_ARGS(x, ...) _NV_REMOVE_FIRST_ARGS1(__VA_ARGS__)

#define _NV_NUM_ARGS(...)  _NV_NUM_ARGS0(__VA_ARGS__)
#define _NV_NUM_ARGS0(...) _NV_EVAL(_NV_NUM_ARGS1(__VA_ARGS__))
#define _NV_NUM_ARGS1(...) _NV_IF(_NV_ISEMPTY(__VA_ARGS__), 0, _NV_NUM_ARGS2(__VA_ARGS__))
#define _NV_NUM_ARGS2(...) \
  _NV_ARG32(               \
    __VA_ARGS__,           \
    31,                    \
    30,                    \
    29,                    \
    28,                    \
    27,                    \
    26,                    \
    25,                    \
    24,                    \
    23,                    \
    22,                    \
    21,                    \
    20,                    \
    19,                    \
    18,                    \
    17,                    \
    16,                    \
    15,                    \
    14,                    \
    13,                    \
    12,                    \
    11,                    \
    10,                    \
    9,                     \
    8,                     \
    7,                     \
    6,                     \
    5,                     \
    4,                     \
    3,                     \
    2,                     \
    1,                     \
    0)

#define _NV_DISPATCH_N_IMPL1(name, ...)        _NV_EVAL(name(__VA_ARGS__))
#define _NV_DISPATCH_N_IMPL0(depth, name, ...) _NV_DISPATCH_N_IMPL1(_NV_CONCAT_EVAL(name, depth), __VA_ARGS__)
#define _NV_DISPATCH_N_IMPL(name, ...)         _NV_DISPATCH_N_IMPL0(_NV_NUM_ARGS(__VA_ARGS__), name, __VA_ARGS__)
#define _NV_DISPATCH_N_ARY(name, ...)          _NV_DISPATCH_N_IMPL(name, __VA_ARGS__)

#endif // __NV_PREPROCESSOR

// Identifies this header as versioned nv target header.
#define __NV_TARGET_VERSIONED

// This guard has 2 purposes - preventing multiple inclusions and allowing to detect whether the header is versioned or
// or not without including the contents.
#if !defined(__NV_TARGET_H)

// Hardcoded version. We cannot use something like __NV_TARGET_VERSION_CURRENT, because there would be collisions.
#define __NV_TARGET_VERSION_03_02 302

// Update the max version. If this is the first include, set it to the current version.
#if !defined(__NV_TARGET_VERSION_MAX)
#  define __NV_TARGET_FIRST_INCLUDE
#  define __NV_TARGET_VERSION_MAX __NV_TARGET_VERSION_03_02
#elif __NV_TARGET_VERSION_MAX <= __NV_TARGET_VERSION_03_02
#  undef __NV_TARGET_VERSION_MAX
#  define __NV_TARGET_VERSION_MAX __NV_TARGET_VERSION_03_02
#endif

// Clear versioned header definition before including next header.
#undef __NV_TARGET_VERSIONED

// Include the next <nv/target> header if available. We include the next header twice - once with __NV_TARGET_H defined
// to determine if the next header is versioned and if so we continue the search. If it is not a versioned header, it
// will be always older than this one and we stop the search.
//
// We use __has_include for the first time to prevent cases when this file is prepended to a .cpp file. It would cause
// errors, because #include_next can be only used inside an included header.
#if defined(__NV_TARGET_FIRST_INCLUDE)
#  undef __NV_TARGET_FIRST_INCLUDE
#  if __has_include(<nv/target>)
#    define __NV_TARGET_H
#    include <nv/target>
#    undef __NV_TARGET_H
#    if defined(__NV_TARGET_VERSIONED)
#      include <nv/target>
#    endif // __NV_TARGET_VERSIONED
#  endif // __has_include(<nv/target>)
#elif __has_include_next(<nv/target>)
#  define __NV_TARGET_H
#  include_next <nv/target>
#  undef __NV_TARGET_H
#  if defined(__NV_TARGET_VERSIONED)
#    include_next <nv/target>
#  endif // __NV_TARGET_VERSIONED
#endif

// If this header version and the max version don't match, skip this include.
#if __NV_TARGET_VERSION_MAX != __NV_TARGET_VERSION_03_02
#  define __NV_SKIP_THIS_INCLUDE
#endif

#ifndef __NV_SKIP_THIS_INCLUDE

#  ifndef __NV_TARGET_H
#    define __NV_TARGET_H

#    if defined(__NVCC__) || defined(__CUDACC_RTC__)
#      define _NV_COMPILER_NVCC
#    elif defined(__NVCOMPILER) && __cplusplus >= 201103L
#      define _NV_COMPILER_NVCXX
#    elif defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
// clang compiling CUDA code, device mode.
#      define _NV_COMPILER_CLANG_CUDA
#    endif

// Hide `if target` support from NVRTC and C
// Some toolkit headers use <nv/target> in true C contexts
#    if !defined(__CUDACC_RTC__) && defined(__cplusplus)

#      if defined(_NV_COMPILER_NVCXX)
#        define _NV_BITSET_ATTRIBUTE [[nv::__target_bitset]]
#      else
#        define _NV_BITSET_ATTRIBUTE
#      endif

namespace nv::target
{
namespace detail
{
typedef unsigned long long base_int_t;

// No host specialization
constexpr base_int_t all_hosts = 1;

// NVIDIA GPUs
constexpr base_int_t sm_35_bit  = base_int_t{1} << 1;
constexpr base_int_t sm_37_bit  = base_int_t{1} << 2;
constexpr base_int_t sm_50_bit  = base_int_t{1} << 3;
constexpr base_int_t sm_52_bit  = base_int_t{1} << 4;
constexpr base_int_t sm_53_bit  = base_int_t{1} << 5;
constexpr base_int_t sm_60_bit  = base_int_t{1} << 6;
constexpr base_int_t sm_61_bit  = base_int_t{1} << 7;
constexpr base_int_t sm_62_bit  = base_int_t{1} << 8;
constexpr base_int_t sm_70_bit  = base_int_t{1} << 9;
constexpr base_int_t sm_72_bit  = base_int_t{1} << 10;
constexpr base_int_t sm_75_bit  = base_int_t{1} << 11;
constexpr base_int_t sm_80_bit  = base_int_t{1} << 12;
constexpr base_int_t sm_86_bit  = base_int_t{1} << 13;
constexpr base_int_t sm_87_bit  = base_int_t{1} << 14;
constexpr base_int_t sm_88_bit  = base_int_t{1} << 15;
constexpr base_int_t sm_89_bit  = base_int_t{1} << 16;
constexpr base_int_t sm_90_bit  = base_int_t{1} << 17;
constexpr base_int_t sm_100_bit = base_int_t{1} << 18;
constexpr base_int_t sm_103_bit = base_int_t{1} << 19;
constexpr base_int_t sm_110_bit = base_int_t{1} << 20;
constexpr base_int_t sm_120_bit = base_int_t{1} << 21;
constexpr base_int_t sm_121_bit = base_int_t{1} << 22;

constexpr base_int_t all_devices =
  base_int_t{0} | sm_35_bit | sm_37_bit | sm_50_bit | sm_52_bit | sm_53_bit | sm_60_bit | sm_61_bit | sm_62_bit
  | sm_70_bit | sm_72_bit | sm_75_bit | sm_80_bit | sm_86_bit | sm_87_bit | sm_88_bit | sm_89_bit | sm_90_bit
  | sm_100_bit | sm_103_bit | sm_110_bit | sm_120_bit | sm_121_bit;

// Store a set of targets as a set of bits
struct _NV_BITSET_ATTRIBUTE target_description
{
  base_int_t targets;

  constexpr target_description(base_int_t a)
      : targets(a)
  {}
};

// The type of the user-visible names of the NVIDIA GPU targets
enum class sm_selector : base_int_t
{
  sm_35  = 35,
  sm_37  = 37,
  sm_50  = 50,
  sm_52  = 52,
  sm_53  = 53,
  sm_60  = 60,
  sm_61  = 61,
  sm_62  = 62,
  sm_70  = 70,
  sm_72  = 72,
  sm_75  = 75,
  sm_80  = 80,
  sm_86  = 86,
  sm_87  = 87,
  sm_88  = 88,
  sm_89  = 89,
  sm_90  = 90,
  sm_100 = 100,
  sm_103 = 103,
  sm_110 = 110,
  sm_120 = 120,
  sm_121 = 121,
};

constexpr base_int_t toint(sm_selector a)
{
  return static_cast<base_int_t>(a);
}

constexpr base_int_t bitexact(sm_selector a)
{
  return toint(a) == 35  ? sm_35_bit
       : toint(a) == 37  ? sm_37_bit
       : toint(a) == 50  ? sm_50_bit
       : toint(a) == 52  ? sm_52_bit
       : toint(a) == 53  ? sm_53_bit
       : toint(a) == 60  ? sm_60_bit
       : toint(a) == 61  ? sm_61_bit
       : toint(a) == 62  ? sm_62_bit
       : toint(a) == 70  ? sm_70_bit
       : toint(a) == 72  ? sm_72_bit
       : toint(a) == 75  ? sm_75_bit
       : toint(a) == 80  ? sm_80_bit
       : toint(a) == 86  ? sm_86_bit
       : toint(a) == 87  ? sm_87_bit
       : toint(a) == 88  ? sm_88_bit
       : toint(a) == 89  ? sm_89_bit
       : toint(a) == 90  ? sm_90_bit
       : toint(a) == 100 ? sm_100_bit
       : toint(a) == 103 ? sm_103_bit
       : toint(a) == 110 ? sm_110_bit
       : toint(a) == 120 ? sm_120_bit
       : toint(a) == 121 ? sm_121_bit
                         : 0;
}

constexpr base_int_t bitrounddown(sm_selector a)
{
  return toint(a) >= 121 ? sm_121_bit
       : toint(a) >= 120 ? sm_120_bit
       : toint(a) >= 110 ? sm_110_bit
       : toint(a) >= 103 ? sm_103_bit
       : toint(a) >= 100 ? sm_100_bit
       : toint(a) >= 90  ? sm_90_bit
       : toint(a) >= 89  ? sm_89_bit
       : toint(a) >= 88  ? sm_88_bit
       : toint(a) >= 87  ? sm_87_bit
       : toint(a) >= 86  ? sm_86_bit
       : toint(a) >= 80  ? sm_80_bit
       : toint(a) >= 75  ? sm_75_bit
       : toint(a) >= 72  ? sm_72_bit
       : toint(a) >= 70  ? sm_70_bit
       : toint(a) >= 62  ? sm_62_bit
       : toint(a) >= 61  ? sm_61_bit
       : toint(a) >= 60  ? sm_60_bit
       : toint(a) >= 53  ? sm_53_bit
       : toint(a) >= 52  ? sm_52_bit
       : toint(a) >= 50  ? sm_50_bit
       : toint(a) >= 37  ? sm_37_bit
       : toint(a) >= 35  ? sm_35_bit
                         : 0;
}

// Public API for NVIDIA GPUs

constexpr target_description is_exactly(sm_selector a)
{
  return target_description(bitexact(a));
}

constexpr target_description provides(sm_selector a)
{
  return target_description(~(bitrounddown(a) - 1) & all_devices);
}

// Boolean operations on target sets

constexpr target_description operator&&(target_description a, target_description b)
{
  return target_description(a.targets & b.targets);
}

constexpr target_description operator||(target_description a, target_description b)
{
  return target_description(a.targets | b.targets);
}

constexpr target_description operator!(target_description a)
{
  return target_description(~a.targets & (all_devices | all_hosts));
}
} // namespace detail

using detail::sm_selector;
using detail::target_description;

// The predicates for basic host/device selection
constexpr target_description is_host    = target_description(detail::all_hosts);
constexpr target_description is_device  = target_description(detail::all_devices);
constexpr target_description any_target = target_description(detail::all_hosts | detail::all_devices);
constexpr target_description no_target  = target_description(0);

// The public names for NVIDIA GPU architectures
constexpr sm_selector sm_35  = sm_selector::sm_35;
constexpr sm_selector sm_37  = sm_selector::sm_37;
constexpr sm_selector sm_50  = sm_selector::sm_50;
constexpr sm_selector sm_52  = sm_selector::sm_52;
constexpr sm_selector sm_53  = sm_selector::sm_53;
constexpr sm_selector sm_60  = sm_selector::sm_60;
constexpr sm_selector sm_61  = sm_selector::sm_61;
constexpr sm_selector sm_62  = sm_selector::sm_62;
constexpr sm_selector sm_70  = sm_selector::sm_70;
constexpr sm_selector sm_72  = sm_selector::sm_72;
constexpr sm_selector sm_75  = sm_selector::sm_75;
constexpr sm_selector sm_80  = sm_selector::sm_80;
constexpr sm_selector sm_86  = sm_selector::sm_86;
constexpr sm_selector sm_87  = sm_selector::sm_87;
constexpr sm_selector sm_88  = sm_selector::sm_88;
constexpr sm_selector sm_89  = sm_selector::sm_89;
constexpr sm_selector sm_90  = sm_selector::sm_90;
constexpr sm_selector sm_100 = sm_selector::sm_100;
constexpr sm_selector sm_103 = sm_selector::sm_103;
constexpr sm_selector sm_110 = sm_selector::sm_110;
constexpr sm_selector sm_120 = sm_selector::sm_120;
constexpr sm_selector sm_121 = sm_selector::sm_121;

using detail::is_exactly;
using detail::provides;
} // namespace nv::target

#    endif // C++  && !defined(__CUDACC_RTC__)

#    define _NV_TARGET_ARCH_TO_SELECTOR_350  nv::target::sm_35
#    define _NV_TARGET_ARCH_TO_SELECTOR_370  nv::target::sm_37
#    define _NV_TARGET_ARCH_TO_SELECTOR_500  nv::target::sm_50
#    define _NV_TARGET_ARCH_TO_SELECTOR_520  nv::target::sm_52
#    define _NV_TARGET_ARCH_TO_SELECTOR_530  nv::target::sm_53
#    define _NV_TARGET_ARCH_TO_SELECTOR_600  nv::target::sm_60
#    define _NV_TARGET_ARCH_TO_SELECTOR_610  nv::target::sm_61
#    define _NV_TARGET_ARCH_TO_SELECTOR_620  nv::target::sm_62
#    define _NV_TARGET_ARCH_TO_SELECTOR_700  nv::target::sm_70
#    define _NV_TARGET_ARCH_TO_SELECTOR_720  nv::target::sm_72
#    define _NV_TARGET_ARCH_TO_SELECTOR_750  nv::target::sm_75
#    define _NV_TARGET_ARCH_TO_SELECTOR_800  nv::target::sm_80
#    define _NV_TARGET_ARCH_TO_SELECTOR_860  nv::target::sm_86
#    define _NV_TARGET_ARCH_TO_SELECTOR_870  nv::target::sm_87
#    define _NV_TARGET_ARCH_TO_SELECTOR_880  nv::target::sm_88
#    define _NV_TARGET_ARCH_TO_SELECTOR_890  nv::target::sm_89
#    define _NV_TARGET_ARCH_TO_SELECTOR_900  nv::target::sm_90
#    define _NV_TARGET_ARCH_TO_SELECTOR_1000 nv::target::sm_100
#    define _NV_TARGET_ARCH_TO_SELECTOR_1030 nv::target::sm_103
#    define _NV_TARGET_ARCH_TO_SELECTOR_1100 nv::target::sm_110
#    define _NV_TARGET_ARCH_TO_SELECTOR_1200 nv::target::sm_120
#    define _NV_TARGET_ARCH_TO_SELECTOR_1210 nv::target::sm_121

#    define _NV_TARGET_ARCH_TO_SM_350  35
#    define _NV_TARGET_ARCH_TO_SM_370  37
#    define _NV_TARGET_ARCH_TO_SM_500  50
#    define _NV_TARGET_ARCH_TO_SM_520  52
#    define _NV_TARGET_ARCH_TO_SM_530  53
#    define _NV_TARGET_ARCH_TO_SM_600  60
#    define _NV_TARGET_ARCH_TO_SM_610  61
#    define _NV_TARGET_ARCH_TO_SM_620  62
#    define _NV_TARGET_ARCH_TO_SM_700  70
#    define _NV_TARGET_ARCH_TO_SM_720  72
#    define _NV_TARGET_ARCH_TO_SM_750  75
#    define _NV_TARGET_ARCH_TO_SM_800  80
#    define _NV_TARGET_ARCH_TO_SM_860  86
#    define _NV_TARGET_ARCH_TO_SM_870  87
#    define _NV_TARGET_ARCH_TO_SM_880  88
#    define _NV_TARGET_ARCH_TO_SM_890  89
#    define _NV_TARGET_ARCH_TO_SM_900  90
#    define _NV_TARGET_ARCH_TO_SM_1000 100
#    define _NV_TARGET_ARCH_TO_SM_1030 103
#    define _NV_TARGET_ARCH_TO_SM_1100 110
#    define _NV_TARGET_ARCH_TO_SM_1200 120
#    define _NV_TARGET_ARCH_TO_SM_1210 121

// Only enable when compiling for CUDA/stdpar
#    if defined(_NV_COMPILER_NVCXX) && defined(_NVHPC_CUDA)

#      define _NV_TARGET_VAL_SM_35  nv::target::sm_35
#      define _NV_TARGET_VAL_SM_37  nv::target::sm_37
#      define _NV_TARGET_VAL_SM_50  nv::target::sm_50
#      define _NV_TARGET_VAL_SM_52  nv::target::sm_52
#      define _NV_TARGET_VAL_SM_53  nv::target::sm_53
#      define _NV_TARGET_VAL_SM_60  nv::target::sm_60
#      define _NV_TARGET_VAL_SM_61  nv::target::sm_61
#      define _NV_TARGET_VAL_SM_62  nv::target::sm_62
#      define _NV_TARGET_VAL_SM_70  nv::target::sm_70
#      define _NV_TARGET_VAL_SM_72  nv::target::sm_72
#      define _NV_TARGET_VAL_SM_75  nv::target::sm_75
#      define _NV_TARGET_VAL_SM_80  nv::target::sm_80
#      define _NV_TARGET_VAL_SM_86  nv::target::sm_86
#      define _NV_TARGET_VAL_SM_87  nv::target::sm_87
#      define _NV_TARGET_VAL_SM_88  nv::target::sm_88
#      define _NV_TARGET_VAL_SM_89  nv::target::sm_89
#      define _NV_TARGET_VAL_SM_90  nv::target::sm_90
#      define _NV_TARGET_VAL_SM_100 nv::target::sm_100
#      define _NV_TARGET_VAL_SM_103 nv::target::sm_103
#      define _NV_TARGET_VAL_SM_110 nv::target::sm_110
#      define _NV_TARGET_VAL_SM_120 nv::target::sm_120
#      define _NV_TARGET_VAL_SM_121 nv::target::sm_121

#      define _NV_TARGET___NV_IS_HOST   nv::target::is_host
#      define _NV_TARGET___NV_IS_DEVICE nv::target::is_device

#      define _NV_TARGET___NV_ANY_TARGET (nv::target::any_target)
#      define _NV_TARGET___NV_NO_TARGET  (nv::target::no_target)

#      if defined(NV_TARGET_SM_INTEGER_LIST)
#        define NV_TARGET_MINIMUM_SM_SELECTOR _NV_FIRST_ARG(NV_TARGET_SM_SELECTOR_LIST)
#        define NV_TARGET_MINIMUM_SM_INTEGER  _NV_FIRST_ARG(NV_TARGET_SM_INTEGER_LIST)
#        define __CUDA_MINIMUM_ARCH__         _NV_CONCAT_EVAL(_NV_FIRST_ARG(NV_TARGET_SM_INTEGER_LIST), 0)
#      endif

#      define _NV_TARGET_PROVIDES(q)   nv::target::provides(q)
#      define _NV_TARGET_IS_EXACTLY(q) nv::target::is_exactly(q)

#    elif defined(_NV_COMPILER_NVCC) || defined(_NV_COMPILER_CLANG_CUDA)

#      define _NV_TARGET_VAL_SM_35  350
#      define _NV_TARGET_VAL_SM_37  370
#      define _NV_TARGET_VAL_SM_50  500
#      define _NV_TARGET_VAL_SM_52  520
#      define _NV_TARGET_VAL_SM_53  530
#      define _NV_TARGET_VAL_SM_60  600
#      define _NV_TARGET_VAL_SM_61  610
#      define _NV_TARGET_VAL_SM_62  620
#      define _NV_TARGET_VAL_SM_70  700
#      define _NV_TARGET_VAL_SM_72  720
#      define _NV_TARGET_VAL_SM_75  750
#      define _NV_TARGET_VAL_SM_80  800
#      define _NV_TARGET_VAL_SM_86  860
#      define _NV_TARGET_VAL_SM_87  870
#      define _NV_TARGET_VAL_SM_88  880
#      define _NV_TARGET_VAL_SM_89  890
#      define _NV_TARGET_VAL_SM_90  900
#      define _NV_TARGET_VAL_SM_100 1000
#      define _NV_TARGET_VAL_SM_103 1030
#      define _NV_TARGET_VAL_SM_110 1100
#      define _NV_TARGET_VAL_SM_120 1200
#      define _NV_TARGET_VAL_SM_121 1210

#      if defined(__CUDA_ARCH__)
#        define _NV_TARGET_VAL                __CUDA_ARCH__
#        define NV_TARGET_MINIMUM_SM_SELECTOR _NV_CONCAT_EVAL(_NV_TARGET_ARCH_TO_SELECTOR_, __CUDA_ARCH__)
#        define NV_TARGET_MINIMUM_SM_INTEGER  _NV_CONCAT_EVAL(_NV_TARGET_ARCH_TO_SM_, __CUDA_ARCH__)
#        define __CUDA_MINIMUM_ARCH__         __CUDA_ARCH__
#      endif

#      if defined(__CUDA_ARCH__)
#        define _NV_TARGET_IS_HOST   0
#        define _NV_TARGET_IS_DEVICE 1
#      else
#        define _NV_TARGET_IS_HOST   1
#        define _NV_TARGET_IS_DEVICE 0
#      endif

#      if defined(_NV_TARGET_VAL)
#        define _NV_DEVICE_CHECK(q) (q)
#      else
#        define _NV_DEVICE_CHECK(q) (0)
#      endif

#      define _NV_TARGET_PROVIDES(q)   _NV_DEVICE_CHECK(_NV_TARGET_VAL >= q)
#      define _NV_TARGET_IS_EXACTLY(q) _NV_DEVICE_CHECK(_NV_TARGET_VAL == q)

// NVCC/NVCXX not being used, only host dispatches allowed
#    else

#      define _NV_COMPILER_NVCC

#      define _NV_TARGET_VAL_SM_35  350
#      define _NV_TARGET_VAL_SM_37  370
#      define _NV_TARGET_VAL_SM_50  500
#      define _NV_TARGET_VAL_SM_52  520
#      define _NV_TARGET_VAL_SM_53  530
#      define _NV_TARGET_VAL_SM_60  600
#      define _NV_TARGET_VAL_SM_61  610
#      define _NV_TARGET_VAL_SM_62  620
#      define _NV_TARGET_VAL_SM_70  700
#      define _NV_TARGET_VAL_SM_72  720
#      define _NV_TARGET_VAL_SM_75  750
#      define _NV_TARGET_VAL_SM_80  800
#      define _NV_TARGET_VAL_SM_86  860
#      define _NV_TARGET_VAL_SM_87  870
#      define _NV_TARGET_VAL_SM_88  880
#      define _NV_TARGET_VAL_SM_89  890
#      define _NV_TARGET_VAL_SM_90  900
#      define _NV_TARGET_VAL_SM_100 1000
#      define _NV_TARGET_VAL_SM_103 1030
#      define _NV_TARGET_VAL_SM_110 1100
#      define _NV_TARGET_VAL_SM_120 1200
#      define _NV_TARGET_VAL_SM_121 1210

#      define _NV_TARGET_VAL 0

#      define _NV_TARGET_IS_HOST   1
#      define _NV_TARGET_IS_DEVICE 0

#      define _NV_DEVICE_CHECK(q) (false)

#      define _NV_TARGET_PROVIDES(q)   _NV_DEVICE_CHECK(_NV_TARGET_VAL >= q)
#      define _NV_TARGET_IS_EXACTLY(q) _NV_DEVICE_CHECK(_NV_TARGET_VAL == q)

#    endif

#    define _NV_TARGET___NV_PROVIDES_SM_35  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_35))
#    define _NV_TARGET___NV_PROVIDES_SM_37  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_37))
#    define _NV_TARGET___NV_PROVIDES_SM_50  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_50))
#    define _NV_TARGET___NV_PROVIDES_SM_52  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_52))
#    define _NV_TARGET___NV_PROVIDES_SM_53  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_53))
#    define _NV_TARGET___NV_PROVIDES_SM_60  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_60))
#    define _NV_TARGET___NV_PROVIDES_SM_61  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_61))
#    define _NV_TARGET___NV_PROVIDES_SM_62  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_62))
#    define _NV_TARGET___NV_PROVIDES_SM_70  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_70))
#    define _NV_TARGET___NV_PROVIDES_SM_72  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_72))
#    define _NV_TARGET___NV_PROVIDES_SM_75  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_75))
#    define _NV_TARGET___NV_PROVIDES_SM_80  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_80))
#    define _NV_TARGET___NV_PROVIDES_SM_86  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_86))
#    define _NV_TARGET___NV_PROVIDES_SM_87  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_87))
#    define _NV_TARGET___NV_PROVIDES_SM_88  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_88))
#    define _NV_TARGET___NV_PROVIDES_SM_89  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_89))
#    define _NV_TARGET___NV_PROVIDES_SM_90  (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_90))
#    define _NV_TARGET___NV_PROVIDES_SM_100 (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_100))
#    define _NV_TARGET___NV_PROVIDES_SM_103 (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_103))
#    define _NV_TARGET___NV_PROVIDES_SM_110 (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_110))
#    define _NV_TARGET___NV_PROVIDES_SM_120 (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_120))
#    define _NV_TARGET___NV_PROVIDES_SM_121 (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_121))

#    define _NV_TARGET___NV_IS_EXACTLY_SM_35  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_35))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_37  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_37))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_50  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_50))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_52  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_52))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_53  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_53))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_60  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_60))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_61  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_61))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_62  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_62))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_70  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_70))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_72  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_72))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_75  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_75))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_80  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_80))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_86  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_86))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_87  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_87))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_88  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_88))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_89  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_89))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_90  (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_90))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_100 (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_100))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_103 (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_103))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_110 (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_110))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_120 (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_120))
#    define _NV_TARGET___NV_IS_EXACTLY_SM_121 (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_121))

#    define NV_PROVIDES_SM_35  __NV_PROVIDES_SM_35
#    define NV_PROVIDES_SM_37  __NV_PROVIDES_SM_37
#    define NV_PROVIDES_SM_50  __NV_PROVIDES_SM_50
#    define NV_PROVIDES_SM_52  __NV_PROVIDES_SM_52
#    define NV_PROVIDES_SM_53  __NV_PROVIDES_SM_53
#    define NV_PROVIDES_SM_60  __NV_PROVIDES_SM_60
#    define NV_PROVIDES_SM_61  __NV_PROVIDES_SM_61
#    define NV_PROVIDES_SM_62  __NV_PROVIDES_SM_62
#    define NV_PROVIDES_SM_70  __NV_PROVIDES_SM_70
#    define NV_PROVIDES_SM_72  __NV_PROVIDES_SM_72
#    define NV_PROVIDES_SM_75  __NV_PROVIDES_SM_75
#    define NV_PROVIDES_SM_80  __NV_PROVIDES_SM_80
#    define NV_PROVIDES_SM_86  __NV_PROVIDES_SM_86
#    define NV_PROVIDES_SM_87  __NV_PROVIDES_SM_87
#    define NV_PROVIDES_SM_88  __NV_PROVIDES_SM_88
#    define NV_PROVIDES_SM_89  __NV_PROVIDES_SM_89
#    define NV_PROVIDES_SM_90  __NV_PROVIDES_SM_90
#    define NV_PROVIDES_SM_100 __NV_PROVIDES_SM_100
#    define NV_PROVIDES_SM_103 __NV_PROVIDES_SM_103
#    define NV_PROVIDES_SM_110 __NV_PROVIDES_SM_110
#    define NV_PROVIDES_SM_120 __NV_PROVIDES_SM_120
#    define NV_PROVIDES_SM_121 __NV_PROVIDES_SM_121

#    define NV_IS_EXACTLY_SM_35  __NV_IS_EXACTLY_SM_35
#    define NV_IS_EXACTLY_SM_37  __NV_IS_EXACTLY_SM_37
#    define NV_IS_EXACTLY_SM_50  __NV_IS_EXACTLY_SM_50
#    define NV_IS_EXACTLY_SM_52  __NV_IS_EXACTLY_SM_52
#    define NV_IS_EXACTLY_SM_53  __NV_IS_EXACTLY_SM_53
#    define NV_IS_EXACTLY_SM_60  __NV_IS_EXACTLY_SM_60
#    define NV_IS_EXACTLY_SM_61  __NV_IS_EXACTLY_SM_61
#    define NV_IS_EXACTLY_SM_62  __NV_IS_EXACTLY_SM_62
#    define NV_IS_EXACTLY_SM_70  __NV_IS_EXACTLY_SM_70
#    define NV_IS_EXACTLY_SM_72  __NV_IS_EXACTLY_SM_72
#    define NV_IS_EXACTLY_SM_75  __NV_IS_EXACTLY_SM_75
#    define NV_IS_EXACTLY_SM_80  __NV_IS_EXACTLY_SM_80
#    define NV_IS_EXACTLY_SM_86  __NV_IS_EXACTLY_SM_86
#    define NV_IS_EXACTLY_SM_87  __NV_IS_EXACTLY_SM_87
#    define NV_IS_EXACTLY_SM_88  __NV_IS_EXACTLY_SM_88
#    define NV_IS_EXACTLY_SM_89  __NV_IS_EXACTLY_SM_89
#    define NV_IS_EXACTLY_SM_90  __NV_IS_EXACTLY_SM_90
#    define NV_IS_EXACTLY_SM_100 __NV_IS_EXACTLY_SM_100
#    define NV_IS_EXACTLY_SM_103 __NV_IS_EXACTLY_SM_103
#    define NV_IS_EXACTLY_SM_110 __NV_IS_EXACTLY_SM_110
#    define NV_IS_EXACTLY_SM_120 __NV_IS_EXACTLY_SM_120
#    define NV_IS_EXACTLY_SM_121 __NV_IS_EXACTLY_SM_121

// Disable SM_90a support on non-supporting compilers.
// Will re-enable for nvcc below.
#    define NV_HAS_FEATURE_SM_90a  NV_NO_TARGET
#    define NV_HAS_FEATURE_SM_100a NV_NO_TARGET
#    define NV_HAS_FEATURE_SM_103a NV_NO_TARGET
#    define NV_HAS_FEATURE_SM_110a NV_NO_TARGET
#    define NV_HAS_FEATURE_SM_120a NV_NO_TARGET
#    define NV_HAS_FEATURE_SM_121a NV_NO_TARGET

#    define NV_HAS_FEATURE_SM_100f NV_NO_TARGET
#    define NV_HAS_FEATURE_SM_103f NV_NO_TARGET
#    define NV_HAS_FEATURE_SM_110f NV_NO_TARGET
#    define NV_HAS_FEATURE_SM_120f NV_NO_TARGET
#    define NV_HAS_FEATURE_SM_121f NV_NO_TARGET

#    define NV_IS_HOST   __NV_IS_HOST
#    define NV_IS_DEVICE __NV_IS_DEVICE

#    define NV_ANY_TARGET __NV_ANY_TARGET
#    define NV_NO_TARGET  __NV_NO_TARGET

// Platform invoke mechanisms
#    if defined(_NV_COMPILER_NVCXX) && defined(_NVHPC_CUDA)

#      define _NV_ARCH_COND(q) (_NV_TARGET_##q)

#      define _NV_BLOCK_EXPAND(...) _NV_REMOVE_PAREN(__VA_ARGS__)

#      define _NV_TARGET_IF(cond, t, ...) \
        (if target _NV_ARCH_COND(cond) { _NV_BLOCK_EXPAND(t) } else {_NV_BLOCK_EXPAND(__VA_ARGS__)})

#    elif defined(_NV_COMPILER_NVCC) || defined(_NV_COMPILER_CLANG_CUDA)

#      if _NV_TARGET___NV_IS_EXACTLY_SM_35
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_35 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_35 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_37
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_37 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_37 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_50
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_50 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_50 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_52
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_52 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_52 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_53
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_53 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_53 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_60
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_60 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_60 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_61
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_61 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_61 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_62
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_62 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_62 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_70
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_70 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_70 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_72
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_72 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_72 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_75
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_75 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_75 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_80
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_80 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_80 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_86
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_86 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_86 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_87
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_87 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_87 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_88
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_88 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_88 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_89
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_89 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_89 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_90
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_90 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_90 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_100
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_100 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_100 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_103
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_103 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_103 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_110
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_110 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_110 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_120
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_120 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_120 0
#      endif

#      if _NV_TARGET___NV_IS_EXACTLY_SM_121
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_121 1
#      else
#        define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_121 0
#      endif

//----------------------------------------------------------------------------------------------------------------------
// architecture-specific SM versions
//----------------------------------------------------------------------------------------------------------------------

// Re-enable sm_90a support in nvcc.
#      undef NV_HAS_FEATURE_SM_90a
#      define NV_HAS_FEATURE_SM_90a __NV_HAS_FEATURE_SM_90a
#      if defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900))
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_90a 1
#      else
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_90a 0
#      endif

// Re-enable sm_100a support in nvcc.
#      undef NV_HAS_FEATURE_SM_100a
#      define NV_HAS_FEATURE_SM_100a __NV_HAS_FEATURE_SM_100a
#      if defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_100a 1
#      else
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_100a 0
#      endif

// Re-enable sm_103a support in nvcc.
#      undef NV_HAS_FEATURE_SM_103a
#      define NV_HAS_FEATURE_SM_103a __NV_HAS_FEATURE_SM_103a
#      if defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_103a 1
#      else
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_103a 0
#      endif

// Re-enable sm_110a support in nvcc.
#      undef NV_HAS_FEATURE_SM_110a
#      define NV_HAS_FEATURE_SM_110a __NV_HAS_FEATURE_SM_110a
#      if defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_110a 1
#      else
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_110a 0
#      endif

// Re-enable sm_120a support in nvcc.
#      undef NV_HAS_FEATURE_SM_120a
#      define NV_HAS_FEATURE_SM_120a __NV_HAS_FEATURE_SM_120a
#      if defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_120a 1
#      else
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_120a 0
#      endif

// Re-enable sm_121a support in nvcc.
#      undef NV_HAS_FEATURE_SM_121a
#      define NV_HAS_FEATURE_SM_121a __NV_HAS_FEATURE_SM_121a
#      if defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_121a 1
#      else
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_121a 0
#      endif

//----------------------------------------------------------------------------------------------------------------------
// family-specific SM versions
//----------------------------------------------------------------------------------------------------------------------

// Re-enable sm_100f support in nvcc.
#      undef NV_HAS_FEATURE_SM_100f
#      define NV_HAS_FEATURE_SM_100f __NV_HAS_FEATURE_SM_100f
#      if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000)
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_100f 1
#      else
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_100f 0
#      endif

// Re-enable sm_103f support in nvcc.
#      undef NV_HAS_FEATURE_SM_103f
#      define NV_HAS_FEATURE_SM_103f __NV_HAS_FEATURE_SM_103f
#      if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030)
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_103f 1
#      else
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_103f 0
#      endif

// Re-enable sm_110f support in nvcc.
#      undef NV_HAS_FEATURE_SM_110f
#      define NV_HAS_FEATURE_SM_110f __NV_HAS_FEATURE_SM_110f
#      if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100)
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_110f 1
#      else
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_110f 0
#      endif

// Re-enable sm_120f support in nvcc.
#      undef NV_HAS_FEATURE_SM_120f
#      define NV_HAS_FEATURE_SM_120f __NV_HAS_FEATURE_SM_120f
#      if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200)
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_120f 1
#      else
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_120f 0
#      endif

// Re-enable sm_121f support in nvcc.
#      undef NV_HAS_FEATURE_SM_121f
#      define NV_HAS_FEATURE_SM_121f __NV_HAS_FEATURE_SM_121f
#      if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210)
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_121f 1
#      else
#        define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_121f 0
#      endif

#      if (_NV_TARGET_IS_HOST)
#        define _NV_TARGET_BOOL___NV_IS_HOST   1
#        define _NV_TARGET_BOOL___NV_IS_DEVICE 0
#      else
#        define _NV_TARGET_BOOL___NV_IS_HOST   0
#        define _NV_TARGET_BOOL___NV_IS_DEVICE 1
#      endif

#      define _NV_TARGET_BOOL___NV_ANY_TARGET 1
#      define _NV_TARGET_BOOL___NV_NO_TARGET  0

// NVCC Greater than stuff

#      if (_NV_TARGET___NV_PROVIDES_SM_35)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_35 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_35 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_37)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_37 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_37 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_50)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_50 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_50 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_52)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_52 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_52 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_53)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_53 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_53 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_60)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_60 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_60 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_61)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_61 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_61 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_62)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_62 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_62 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_70)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_70 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_70 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_72)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_72 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_72 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_75)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_75 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_75 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_80)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_80 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_80 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_86)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_86 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_86 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_87)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_87 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_87 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_88)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_88 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_88 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_89)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_89 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_89 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_90)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_90 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_90 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_100)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_100 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_100 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_103)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_103 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_103 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_110)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_110 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_110 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_120)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_120 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_120 0
#      endif

#      if (_NV_TARGET___NV_PROVIDES_SM_121)
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_121 1
#      else
#        define _NV_TARGET_BOOL___NV_PROVIDES_SM_121 0
#      endif

#      define _NV_ARCH_COND_CAT1(cond) _NV_TARGET_BOOL_##cond
#      define _NV_ARCH_COND_CAT(cond)  _NV_EVAL(_NV_ARCH_COND_CAT1(cond))

#      define _NV_TARGET_EMPTY_PARAM ;

#      define _NV_BLOCK_EXPAND(...)       {_NV_REMOVE_PAREN(__VA_ARGS__)}
#      define _NV_TARGET_IF(cond, t, ...) _NV_IF(_NV_ARCH_COND_CAT(cond), t, __VA_ARGS__)

#    endif // _NV_COMPILER_NVCC

#    define _NV_TARGET_DISPATCH_HANDLE0()
#    define _NV_TARGET_DISPATCH_HANDLE2(q, fn)       _NV_TARGET_IF(q, fn)
#    define _NV_TARGET_DISPATCH_HANDLE4(q, fn, ...)  _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE2(__VA_ARGS__))
#    define _NV_TARGET_DISPATCH_HANDLE6(q, fn, ...)  _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE4(__VA_ARGS__))
#    define _NV_TARGET_DISPATCH_HANDLE8(q, fn, ...)  _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE6(__VA_ARGS__))
#    define _NV_TARGET_DISPATCH_HANDLE10(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE8(__VA_ARGS__))
#    define _NV_TARGET_DISPATCH_HANDLE12(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE10(__VA_ARGS__))
#    define _NV_TARGET_DISPATCH_HANDLE14(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE12(__VA_ARGS__))
#    define _NV_TARGET_DISPATCH_HANDLE16(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE14(__VA_ARGS__))
#    define _NV_TARGET_DISPATCH_HANDLE18(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE16(__VA_ARGS__))
#    define _NV_TARGET_DISPATCH_HANDLE20(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE18(__VA_ARGS__))
#    define _NV_TARGET_DISPATCH_HANDLE22(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE20(__VA_ARGS__))
#    define _NV_TARGET_DISPATCH_HANDLE24(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE22(__VA_ARGS__))
#    define _NV_TARGET_DISPATCH_HANDLE26(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE24(__VA_ARGS__))
#    define _NV_TARGET_DISPATCH_HANDLE28(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE26(__VA_ARGS__))
#    define _NV_TARGET_DISPATCH_HANDLE30(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE28(__VA_ARGS__))
#    define _NV_TARGET_DISPATCH_HANDLE32(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE30(__VA_ARGS__))
#    define _NV_TARGET_DISPATCH_HANDLE34(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE32(__VA_ARGS__))

#    define _NV_TARGET_DISPATCH(...) _NV_BLOCK_EXPAND(_NV_DISPATCH_N_ARY(_NV_TARGET_DISPATCH_HANDLE, __VA_ARGS__))

// NV_IF_TARGET supports a false statement provided as a variadic macro
#    define NV_IF_TARGET(cond, ...)       _NV_BLOCK_EXPAND(_NV_TARGET_IF(cond, __VA_ARGS__))
#    define NV_IF_ELSE_TARGET(cond, t, f) _NV_BLOCK_EXPAND(_NV_TARGET_IF(cond, t, f))
#    define NV_DISPATCH_TARGET(...)       _NV_TARGET_DISPATCH(__VA_ARGS__)

#  endif // __NV_TARGET_H

#else
#  undef __NV_SKIP_THIS_INCLUDE
#endif

#endif // __NV_TARGET_H
