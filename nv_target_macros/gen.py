#!/usr/bin/env python3

import argparse
from datetime import datetime

ARCH_LIST = [
    35,
    37,
    50,
    52,
    53,
    60,
    61,
    62,
    70,
    72,
    75,
    80,
    86,
    87,
    88,
    89,
    90,
    100,
    103,
    110,
    120,
    121,
]
ARCH_LIST_A = list(filter(lambda arch: arch >= 90, ARCH_LIST))
ARCH_LIST_F = list(filter(lambda arch: arch >= 100, ARCH_LIST))


def preprocessor_contents() -> str:
    return """\
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
#define _NV_ARG32_0( \\
  _0,                \\
  _1,                \\
  _2,                \\
  _3,                \\
  _4,                \\
  _5,                \\
  _6,                \\
  _7,                \\
  _8,                \\
  _9,                \\
  _10,               \\
  _11,               \\
  _12,               \\
  _13,               \\
  _14,               \\
  _15,               \\
  _16,               \\
  _17,               \\
  _18,               \\
  _19,               \\
  _20,               \\
  _21,               \\
  _22,               \\
  _23,               \\
  _24,               \\
  _25,               \\
  _26,               \\
  _27,               \\
  _28,               \\
  _29,               \\
  _30,               \\
  _31,               \\
  ...)               \\
  _31

#define _NV_HAS_COMMA(...) \\
  _NV_ARG32(__VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0)

#define _NV_TRIGGER_PARENTHESIS_(...) ,

/*
This tests a variety of conditions for determining what the incoming statement is.
1. test if there is just one argument
2. test if _TRIGGER_PARENTHESIS_ together with the argument adds a comma
3. test if the argument together with a parenthesis adds a comma
4. test if placing it between _TRIGGER_PARENTHESIS_ and the parenthesis adds a comma
*/
#define _NV_ISEMPTY(...)                                                      \\
  _NV_ISEMPTY0(_NV_EVAL(_NV_HAS_COMMA(__VA_ARGS__)),                          \\
               _NV_EVAL(_NV_HAS_COMMA(_NV_TRIGGER_PARENTHESIS_ __VA_ARGS__)), \\
               _NV_EVAL(_NV_HAS_COMMA(__VA_ARGS__(/*empty*/))),               \\
               _NV_EVAL(_NV_HAS_COMMA(_NV_TRIGGER_PARENTHESIS_ __VA_ARGS__(/*empty*/))))

#define _NV_PASTE5(_0, _1, _2, _3, _4) _0##_1##_2##_3##_4
#define _NV_ISEMPTY0(_0, _1, _2, _3)   _NV_HAS_COMMA(_NV_PASTE5(_NV_IS_EMPTY_CASE_, _0, _1, _2, _3))
#define _NV_IS_EMPTY_CASE_0001         ,

#define _NV_REMOVE_PAREN(...) _NV_REMOVE_PAREN1(__VA_ARGS__)
#define _NV_REMOVE_PAREN1(...) \\
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
#define _NV_NUM_ARGS2(...) \\
  _NV_ARG32(               \\
    __VA_ARGS__,           \\
    31,                    \\
    30,                    \\
    29,                    \\
    28,                    \\
    27,                    \\
    26,                    \\
    25,                    \\
    24,                    \\
    23,                    \\
    22,                    \\
    21,                    \\
    20,                    \\
    19,                    \\
    18,                    \\
    17,                    \\
    16,                    \\
    15,                    \\
    14,                    \\
    13,                    \\
    12,                    \\
    11,                    \\
    10,                    \\
    9,                     \\
    8,                     \\
    7,                     \\
    6,                     \\
    5,                     \\
    4,                     \\
    3,                     \\
    2,                     \\
    1,                     \\
    0)

#define _NV_DISPATCH_N_IMPL1(name, ...)        _NV_EVAL(name(__VA_ARGS__))
#define _NV_DISPATCH_N_IMPL0(depth, name, ...) _NV_DISPATCH_N_IMPL1(_NV_CONCAT_EVAL(name, depth), __VA_ARGS__)
#define _NV_DISPATCH_N_IMPL(name, ...)         _NV_DISPATCH_N_IMPL0(_NV_NUM_ARGS(__VA_ARGS__), name, __VA_ARGS__)
#define _NV_DISPATCH_N_ARY(name, ...)          _NV_DISPATCH_N_IMPL(name, __VA_ARGS__)

#endif // __NV_PREPROCESSOR
"""


def target_contents() -> str:
    def make_sm_arch_bits() -> str:
        return "".join(
            f"constexpr base_int_t sm_{arch}_bit = base_int_t{{1}} << {i + 1};\n"
            for i, arch in enumerate(ARCH_LIST)
        )

    def make_all_devices() -> str:
        return "".join(f"\n  | sm_{arch}_bit" for arch in ARCH_LIST)

    def make_sm_selector_vals() -> str:
        return "".join(f"\n  sm_{arch} = {arch}," for arch in ARCH_LIST)

    def make_bit_exact(var_name: str) -> str:
        return "".join(
            f"\n    toint({var_name}) == {arch} ? sm_{arch}_bit :" for arch in ARCH_LIST
        )

    def make_bitrounddown(var_name: str) -> str:
        return "".join(
            f"\n    toint({var_name}) >= {arch} ? sm_{arch}_bit :"
            for arch in reversed(ARCH_LIST)
        )

    def make_public_names() -> str:
        return "".join(
            f"constexpr sm_selector sm_{arch} = sm_selector::sm_{arch};\n"
            for arch in ARCH_LIST
        )

    return f"""\
#if defined(__NVCC__) || defined(__CUDACC_RTC__)
#  define _NV_COMPILER_NVCC
#elif defined(__NVCOMPILER) && __cplusplus >= 201103L
#  define _NV_COMPILER_NVCXX
#elif defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
// clang compiling CUDA code, device mode.
#  define _NV_COMPILER_CLANG_CUDA
#endif

// Hide `if target` support from NVRTC and C
// Some toolkit headers use <nv/target> in true C contexts
#if !defined(__CUDACC_RTC__) && defined(__cplusplus)

#  if defined(_NV_COMPILER_NVCXX)
#    define _NV_BITSET_ATTRIBUTE [[nv::__target_bitset]]
#  else
#    define _NV_BITSET_ATTRIBUTE
#  endif

namespace nv::target
{{
namespace detail
{{
typedef unsigned long long base_int_t;

// No host specialization
constexpr base_int_t all_hosts = 1;

// NVIDIA GPUs
{make_sm_arch_bits()}
constexpr base_int_t all_devices = base_int_t{{0}}{make_all_devices()};

// Store a set of targets as a set of bits
struct _NV_BITSET_ATTRIBUTE target_description
{{
  base_int_t targets;

  constexpr target_description(base_int_t a)
      : targets(a)
  {{}}
}};

// The type of the user-visible names of the NVIDIA GPU targets
enum class sm_selector : base_int_t
{{{make_sm_selector_vals()}
}};

constexpr base_int_t toint(sm_selector a)
{{
  return static_cast<base_int_t>(a);
}}

constexpr base_int_t bitexact(sm_selector a)
{{
  return {make_bit_exact("a")} 0;
}}

constexpr base_int_t bitrounddown(sm_selector a)
{{
  return {make_bitrounddown("a")} 0;
}}

// Public API for NVIDIA GPUs

constexpr target_description is_exactly(sm_selector a)
{{
  return target_description(bitexact(a));
}}

constexpr target_description provides(sm_selector a)
{{
  return target_description(~(bitrounddown(a) - 1) & all_devices);
}}

// Boolean operations on target sets

constexpr target_description operator&&(target_description a, target_description b)
{{
  return target_description(a.targets & b.targets);
}}

constexpr target_description operator||(target_description a, target_description b)
{{
  return target_description(a.targets | b.targets);
}}

constexpr target_description operator!(target_description a)
{{
  return target_description(~a.targets & (all_devices | all_hosts));
}}
}} // namespace detail

using detail::sm_selector;
using detail::target_description;

// The predicates for basic host/device selection
constexpr target_description is_host    = target_description(detail::all_hosts);
constexpr target_description is_device  = target_description(detail::all_devices);
constexpr target_description any_target = target_description(detail::all_hosts | detail::all_devices);
constexpr target_description no_target  = target_description(0);

// The public names for NVIDIA GPU architectures
{make_public_names()}

using detail::is_exactly;
using detail::provides;
}} // namespace nv::target

#endif // C++  && !defined(__CUDACC_RTC__)
"""


def target_macros_contents() -> str:
    def make_arch_to_selectors() -> str:
        return "".join(
            f"#define _NV_TARGET_ARCH_TO_SELECTOR_{arch}0 nv::target::sm_{arch}\n"
            for arch in ARCH_LIST
        )

    def make_arch_to_sms() -> str:
        return "".join(
            f"#define _NV_TARGET_ARCH_TO_SM_{arch}0 {arch}\n" for arch in ARCH_LIST
        )

    def make_val_sms_nvcxx() -> str:
        return "".join(
            f"#define _NV_TARGET_VAL_SM_{arch} nv::target::sm_{arch}\n"
            for arch in ARCH_LIST
        )

    def make_val_sms_nvcc_clang() -> str:
        return "".join(
            f"#define _NV_TARGET_VAL_SM_{arch} {arch}0\n" for arch in ARCH_LIST
        )

    def make_provides_internal() -> str:
        return "".join(
            f"#define _NV_TARGET___NV_PROVIDES_SM_{arch} (_NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_{arch}))\n"
            for arch in ARCH_LIST
        )

    def make_is_exaclty_internal() -> str:
        return "".join(
            f"#define _NV_TARGET___NV_IS_EXACTLY_SM_{arch} (_NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_{arch}))\n"
            for arch in ARCH_LIST
        )

    def make_provides() -> str:
        return "".join(
            f"#define NV_PROVIDES_SM_{arch} __NV_PROVIDES_SM_{arch}\n"
            for arch in ARCH_LIST
        )

    def make_is_exaclty() -> str:
        return "".join(
            f"#define NV_IS_EXACTLY_SM_{arch} __NV_IS_EXACTLY_SM_{arch}\n"
            for arch in ARCH_LIST
        )

    def make_has_feature_arch_a_disabled() -> str:
        return "".join(
            f"#define NV_HAS_FEATURE_SM_{arch}a NV_NO_TARGET\n" for arch in ARCH_LIST_A
        )

    def make_has_feature_arch_f_disabled() -> str:
        return "".join(
            f"#define NV_HAS_FEATURE_SM_{arch}f NV_NO_TARGET\n" for arch in ARCH_LIST_F
        )

    def make_bool_is_exactly_sms_nvcc_clang() -> str:
        return "".join(
            f"""\
#  if _NV_TARGET___NV_IS_EXACTLY_SM_{arch}
#    define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_{arch} 1
#  else
#    define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_{arch} 0
#  endif\n\n"""
            for arch in ARCH_LIST
        )

    def make_has_feature_sms_a_nvcc() -> str:
        return "".join(
            f"""\
// Re-enable sm_{arch}a support in nvcc.
#  undef NV_HAS_FEATURE_SM_{arch}a
#  define NV_HAS_FEATURE_SM_{arch}a __NV_HAS_FEATURE_SM_{arch}a
#  if defined(__CUDA_ARCH_FEAT_SM{arch}_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == {arch}0))
#    define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_{arch}a 1
#  else
#    define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_{arch}a 0
#  endif\n\n"""
            for arch in ARCH_LIST_A
        )

    def make_has_feature_sms_f_nvcc() -> str:
        return "".join(
            f"""\
// Re-enable sm_{arch}f support in nvcc.
#  undef NV_HAS_FEATURE_SM_{arch}f
#  define NV_HAS_FEATURE_SM_{arch}f __NV_HAS_FEATURE_SM_{arch}f
#  if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == {arch}0)
#    define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_{arch}f 1
#  else
#    define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_{arch}f 0
#  endif\n\n"""
            for arch in ARCH_LIST_F
        )

    def make_bool_provides_sms_nvcc_clang() -> str:
        return "".join(
            f"""\
#  if (_NV_TARGET___NV_PROVIDES_SM_{arch})
#    define _NV_TARGET_BOOL___NV_PROVIDES_SM_{arch} 1
#  else
#    define _NV_TARGET_BOOL___NV_PROVIDES_SM_{arch} 0
#  endif\n\n"""
            for arch in ARCH_LIST
        )

    return f"""\
{make_arch_to_selectors()}

{make_arch_to_sms()}

// Only enable when compiling for CUDA/stdpar
#if defined(_NV_COMPILER_NVCXX) && defined(_NVHPC_CUDA)

{make_val_sms_nvcxx()}

#  define _NV_TARGET___NV_IS_HOST   nv::target::is_host
#  define _NV_TARGET___NV_IS_DEVICE nv::target::is_device

#  define _NV_TARGET___NV_ANY_TARGET (nv::target::any_target)
#  define _NV_TARGET___NV_NO_TARGET  (nv::target::no_target)

#  if defined(NV_TARGET_SM_INTEGER_LIST)
#    define NV_TARGET_MINIMUM_SM_SELECTOR _NV_FIRST_ARG(NV_TARGET_SM_SELECTOR_LIST)
#    define NV_TARGET_MINIMUM_SM_INTEGER  _NV_FIRST_ARG(NV_TARGET_SM_INTEGER_LIST)
#    define __CUDA_MINIMUM_ARCH__         _NV_CONCAT_EVAL(_NV_FIRST_ARG(NV_TARGET_SM_INTEGER_LIST), 0)
#  endif

#  define _NV_TARGET_PROVIDES(q)   nv::target::provides(q)
#  define _NV_TARGET_IS_EXACTLY(q) nv::target::is_exactly(q)

#elif defined(_NV_COMPILER_NVCC) || defined(_NV_COMPILER_CLANG_CUDA)

{make_val_sms_nvcc_clang()}

#  if defined(__CUDA_ARCH__)
#    define _NV_TARGET_VAL                __CUDA_ARCH__
#    define NV_TARGET_MINIMUM_SM_SELECTOR _NV_CONCAT_EVAL(_NV_TARGET_ARCH_TO_SELECTOR_, __CUDA_ARCH__)
#    define NV_TARGET_MINIMUM_SM_INTEGER  _NV_CONCAT_EVAL(_NV_TARGET_ARCH_TO_SM_, __CUDA_ARCH__)
#    define __CUDA_MINIMUM_ARCH__         __CUDA_ARCH__
#  endif

#  if defined(__CUDA_ARCH__)
#    define _NV_TARGET_IS_HOST   0
#    define _NV_TARGET_IS_DEVICE 1
#  else
#    define _NV_TARGET_IS_HOST   1
#    define _NV_TARGET_IS_DEVICE 0
#  endif

#  if defined(_NV_TARGET_VAL)
#    define _NV_DEVICE_CHECK(q) (q)
#  else
#    define _NV_DEVICE_CHECK(q) (0)
#  endif

#  define _NV_TARGET_PROVIDES(q)   _NV_DEVICE_CHECK(_NV_TARGET_VAL >= q)
#  define _NV_TARGET_IS_EXACTLY(q) _NV_DEVICE_CHECK(_NV_TARGET_VAL == q)

// NVCC/NVCXX not being used, only host dispatches allowed
#else

#  define _NV_COMPILER_NVCC

{make_val_sms_nvcc_clang()}

#  define _NV_TARGET_VAL 0

#  define _NV_TARGET_IS_HOST   1
#  define _NV_TARGET_IS_DEVICE 0

#  define _NV_DEVICE_CHECK(q) (false)

#  define _NV_TARGET_PROVIDES(q)   _NV_DEVICE_CHECK(_NV_TARGET_VAL >= q)
#  define _NV_TARGET_IS_EXACTLY(q) _NV_DEVICE_CHECK(_NV_TARGET_VAL == q)

#endif

{make_provides_internal()}

{make_is_exaclty_internal()}

{make_provides()}

{make_is_exaclty()}

// Disable SM_90a support on non-supporting compilers.
// Will re-enable for nvcc below.
{make_has_feature_arch_a_disabled()}

{make_has_feature_arch_f_disabled()}

#define NV_IS_HOST   __NV_IS_HOST
#define NV_IS_DEVICE __NV_IS_DEVICE

#define NV_ANY_TARGET __NV_ANY_TARGET
#define NV_NO_TARGET  __NV_NO_TARGET

// Platform invoke mechanisms
#if defined(_NV_COMPILER_NVCXX) && defined(_NVHPC_CUDA)

#  define _NV_ARCH_COND(q) (_NV_TARGET_##q)

#  define _NV_BLOCK_EXPAND(...) _NV_REMOVE_PAREN(__VA_ARGS__)

#  define _NV_TARGET_IF(cond, t, ...) \
    (if target _NV_ARCH_COND(cond) {{ _NV_BLOCK_EXPAND(t) }} else {{_NV_BLOCK_EXPAND(__VA_ARGS__)}})

#elif defined(_NV_COMPILER_NVCC) || defined(_NV_COMPILER_CLANG_CUDA)

{make_bool_is_exactly_sms_nvcc_clang()}

//----------------------------------------------------------------------------------------------------------------------
// architecture-specific SM versions
//----------------------------------------------------------------------------------------------------------------------

{make_has_feature_sms_a_nvcc()}

//----------------------------------------------------------------------------------------------------------------------
// family-specific SM versions
//----------------------------------------------------------------------------------------------------------------------

{make_has_feature_sms_f_nvcc()}

#  if (_NV_TARGET_IS_HOST)
#    define _NV_TARGET_BOOL___NV_IS_HOST   1
#    define _NV_TARGET_BOOL___NV_IS_DEVICE 0
#  else
#    define _NV_TARGET_BOOL___NV_IS_HOST   0
#    define _NV_TARGET_BOOL___NV_IS_DEVICE 1
#  endif

#  define _NV_TARGET_BOOL___NV_ANY_TARGET 1
#  define _NV_TARGET_BOOL___NV_NO_TARGET  0

// NVCC Greater than stuff

{make_bool_provides_sms_nvcc_clang()}

#  define _NV_ARCH_COND_CAT1(cond) _NV_TARGET_BOOL_##cond
#  define _NV_ARCH_COND_CAT(cond)  _NV_EVAL(_NV_ARCH_COND_CAT1(cond))

#  define _NV_TARGET_EMPTY_PARAM ;

#  define _NV_BLOCK_EXPAND(...)       {{_NV_REMOVE_PAREN(__VA_ARGS__)}}
#  define _NV_TARGET_IF(cond, t, ...) _NV_IF(_NV_ARCH_COND_CAT(cond), t, __VA_ARGS__)

#endif // _NV_COMPILER_NVCC

#define _NV_TARGET_DISPATCH_HANDLE0()
#define _NV_TARGET_DISPATCH_HANDLE2(q, fn)       _NV_TARGET_IF(q, fn)
#define _NV_TARGET_DISPATCH_HANDLE4(q, fn, ...)  _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE2(__VA_ARGS__))
#define _NV_TARGET_DISPATCH_HANDLE6(q, fn, ...)  _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE4(__VA_ARGS__))
#define _NV_TARGET_DISPATCH_HANDLE8(q, fn, ...)  _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE6(__VA_ARGS__))
#define _NV_TARGET_DISPATCH_HANDLE10(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE8(__VA_ARGS__))
#define _NV_TARGET_DISPATCH_HANDLE12(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE10(__VA_ARGS__))
#define _NV_TARGET_DISPATCH_HANDLE14(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE12(__VA_ARGS__))
#define _NV_TARGET_DISPATCH_HANDLE16(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE14(__VA_ARGS__))
#define _NV_TARGET_DISPATCH_HANDLE18(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE16(__VA_ARGS__))
#define _NV_TARGET_DISPATCH_HANDLE20(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE18(__VA_ARGS__))
#define _NV_TARGET_DISPATCH_HANDLE22(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE20(__VA_ARGS__))
#define _NV_TARGET_DISPATCH_HANDLE24(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE22(__VA_ARGS__))
#define _NV_TARGET_DISPATCH_HANDLE26(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE24(__VA_ARGS__))
#define _NV_TARGET_DISPATCH_HANDLE28(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE26(__VA_ARGS__))
#define _NV_TARGET_DISPATCH_HANDLE30(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE28(__VA_ARGS__))
#define _NV_TARGET_DISPATCH_HANDLE32(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE30(__VA_ARGS__))
#define _NV_TARGET_DISPATCH_HANDLE34(q, fn, ...) _NV_TARGET_IF(q, fn, _NV_TARGET_DISPATCH_HANDLE32(__VA_ARGS__))

#define _NV_TARGET_DISPATCH(...) _NV_BLOCK_EXPAND(_NV_DISPATCH_N_ARY(_NV_TARGET_DISPATCH_HANDLE, __VA_ARGS__))

// NV_IF_TARGET supports a false statement provided as a variadic macro
#define NV_IF_TARGET(cond, ...)       _NV_BLOCK_EXPAND(_NV_TARGET_IF(cond, __VA_ARGS__))
#define NV_IF_ELSE_TARGET(cond, t, f) _NV_BLOCK_EXPAND(_NV_TARGET_IF(cond, t, f))
#define NV_DISPATCH_TARGET(...)       _NV_TARGET_DISPATCH(__VA_ARGS__)
"""


def main(version_major: int, version_minor: int, filename: int):
    contents = f"""\
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) {datetime.now().year} NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// This header contains a preview of a portability system that enables
// CUDA C++ development with NVC++, NVCC, and supported host compilers.
// These interfaces are not guaranteed to be stable.

{preprocessor_contents()}

// Identifies this header as versioned nv target header.
#define __NV_TARGET_VERSIONED

// This guard has 2 purposes - preventing multiple inclusions and allowing to detect whether the header is versioned or
// or not without including the contents.
#if !defined(__NV_TARGET_H)

// Hardcoded version. We cannot use something like __NV_TARGET_VERSION_CURRENT, because there would be collisions.
#  define __NV_TARGET_VERSION_{version_major:02d}_{version_minor:02d} {version_major * 100 + version_minor}

// Update the max version. If this is the first include, set it to the current version.
#  if !defined(__NV_TARGET_VERSION_MAX)
#    define __NV_TARGET_FIRST_INCLUDE
#    define __NV_TARGET_VERSION_MAX __NV_TARGET_VERSION_{version_major:02d}_{version_minor:02d}
#  elif __NV_TARGET_VERSION_MAX <= __NV_TARGET_VERSION_{version_major:02d}_{version_minor:02d}
#    undef __NV_TARGET_VERSION_MAX
#    define __NV_TARGET_VERSION_MAX __NV_TARGET_VERSION_{version_major:02d}_{version_minor:02d}
#  endif

// Clear versioned header definition before including next header.
#  undef __NV_TARGET_VERSIONED

// Include the next <nv/target> header if available. We include the next header twice - once with __NV_TARGET_H defined
// to determine if the next header is versioned and if so we continue the search. If it is not a versioned header, it
// will be always older than this one and we stop the search.
//
// We use __has_include for the first time to prevent cases when this file is prepended to a .cpp file. It would cause
// errors, because #include_next can be only used inside an included header.
#  if defined(__NV_TARGET_FIRST_INCLUDE)
#    undef __NV_TARGET_FIRST_INCLUDE
#    if _NV_HAS_INCLUDE(<nv/target>)
#      define __NV_TARGET_H
#      include <nv/target>
#      undef __NV_TARGET_H
#      if defined(__NV_TARGET_VERSIONED)
#        include <nv/target>
#      endif // __NV_TARGET_VERSIONED
#    endif // _NV_HAS_INCLUDE(<nv/target>)
#  elif _NV_HAS_INCLUDE_NEXT(<nv/target>)
#    define __NV_TARGET_H
#    include_next <nv/target>
#    undef __NV_TARGET_H
#    if defined(__NV_TARGET_VERSIONED)
#      include_next <nv/target>
#    endif // __NV_TARGET_VERSIONED
#  endif

// If this header version and the max version don't match, skip this include.
#  if __NV_TARGET_VERSION_MAX != __NV_TARGET_VERSION_{version_major:02d}_{version_minor:02d}
#    define __NV_SKIP_THIS_INCLUDE
#  endif

#  ifndef __NV_SKIP_THIS_INCLUDE

#    ifndef __NV_TARGET_H
#    define __NV_TARGET_H

{target_contents()}

{target_macros_contents()}

#    endif // __NV_TARGET_H

#  else
#    undef __NV_SKIP_THIS_INCLUDE
#  endif

#endif // __NV_TARGET_H
"""
    with open(filename, "w") as f:
        f.write(contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("version_major", type=int)
    parser.add_argument("version_minor", type=int)
    args = parser.parse_args()

    main(args.version_major, args.version_minor, args.output)
