//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD__FORMAT_FORMAT_SPEC_PARSER_H
#define _CUDA_STD__FORMAT_FORMAT_SPEC_PARSER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/copy_n.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__concepts/arithmetic.h>
#include <cuda/std/__format/format_arg.h>
#include <cuda/std/__format/format_error.h>
#include <cuda/std/__format/format_parse_context.h>
#include <cuda/std/__format/parse_arg_id.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/underlying_type.h>
#include <cuda/std/__utility/monostate.h>
#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/cstdint>
#include <cuda/std/string_view>

#if !_CCCL_COMPILER(NVRTC)
#  include <string>
#endif // !_CCCL_COMPILER(NVRTC)

// This file contains the std-format-spec parser.
//
// Most of the code can be reused in the chrono-format-spec.
// This header has some support for the chrono-format-spec since it doesn't
// affect the std-format-spec.

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <size_t _IdSize, size_t _OptSize>
[[noreturn]] _CCCL_API inline void
__throw_invalid_option_format_error(const char (&__id)[_IdSize], const char (&__option)[_OptSize])
{
  constexpr string_view __msg_pt1 = "The format specifier for ";
  constexpr string_view __msg_pt2 = " does not allow the ";
  constexpr string_view __msg_pt3 = " option";

  char __msg[__msg_pt1.size() + _IdSize + __msg_pt2.size() + _OptSize + __msg_pt3.size() + 1]{};
  char* __msg_it = __msg;

  const string_view __id_view{__id};
  const string_view __opt_view{__option};

  // copy the message parts into the message buffer
  char_traits<char>::copy(__msg_it, __msg_pt1.data(), __msg_pt1.size());
  char_traits<char>::copy(__msg_it += __msg_pt1.size(), __id_view.data(), __id_view.size());
  char_traits<char>::copy(__msg_it += __id_view.size(), __msg_pt2.data(), __msg_pt2.size());
  char_traits<char>::copy(__msg_it += __msg_pt2.size(), __opt_view.data(), __opt_view.size());
  char_traits<char>::copy(__msg_it += __opt_view.size(), __msg_pt3.data(), __msg_pt3.size());
  __msg_it[__msg_pt3.size()] = '\0';

  ::cuda::std::__throw_format_error(__msg);
}

template <size_t _IdSize>
[[noreturn]] _CCCL_API inline void __throw_invalid_type_format_error(const char (&__id)[_IdSize])
{
  constexpr string_view __msg_pt1 = "The type option contains an invalid value for ";
  constexpr string_view __msg_pt2 = " formatting argument";

  char __msg[__msg_pt1.size() + _IdSize + __msg_pt2.size() + 1]{};
  char* __msg_it = __msg;

  const string_view __id_view{__id};

  // copy the message parts into the message buffer
  char_traits<char>::copy(__msg_it, __msg_pt1.data(), __msg_pt1.size());
  char_traits<char>::copy(__msg_it += __msg_pt1.size(), __id_view.data(), __id_view.size());
  char_traits<char>::copy(__msg_it += __id_view.size(), __msg_pt2.data(), __msg_pt2.size());
  __msg_it[__msg_pt2.size()] = '\0';

  ::cuda::std::__throw_format_error(__msg);
}

struct __fmt_substitute_arg_id_visitor
{
  template <class _Tp>
  [[nodiscard]] _CCCL_API constexpr uint32_t operator()([[maybe_unused]] _Tp __arg)
  {
    if constexpr (is_same_v<_Tp, monostate>)
    {
      ::cuda::std::__throw_format_error("The argument index value is too large for the number of arguments supplied");
    }

    // [format.string.std]/8
    // If { arg-idopt } is used in a width or precision, the value of the
    // corresponding formatting argument is used in its place. If the
    // corresponding formatting argument is not of standard signed or unsigned
    // integer type, or its value is negative for precision or non-positive for
    // width, an exception of type format_error is thrown.
    //
    // When an integral is used in a format function, it is stored as one of
    // the types checked below. Other integral types are promoted. For example,
    // a signed char is stored as an int.
    else if constexpr (is_same_v<_Tp, int> || is_same_v<_Tp, unsigned int> || //
                       is_same_v<_Tp, long long> || is_same_v<_Tp, unsigned long long>)
    {
      if constexpr (is_signed_v<_Tp>)
      {
        if (__arg < 0)
        {
          ::cuda::std::__throw_format_error("An argument index may not have a negative value");
        }
      }

      using _CT = common_type_t<_Tp, decltype(__fmt_number_max)>;
      if (static_cast<_CT>(__arg) > static_cast<_CT>(__fmt_number_max))
      {
        ::cuda::std::__throw_format_error("The value of the argument index exceeds its maximum value");
      }

      return static_cast<uint32_t>(__arg);
    }
    else
    {
      ::cuda::std::__throw_format_error("Replacement argument isn't a standard signed or unsigned integer type");
    }
  }
};

template <class _Context>
[[nodiscard]] _CCCL_API constexpr uint32_t __fmt_substitute_arg_id(basic_format_arg<_Context> __format_arg)
{
  // [format.string.std]/8
  //   If the corresponding formatting argument is not of integral type...
  // This wording allows char and bool too. LWG-3720 changes the wording to
  //    If the corresponding formatting argument is not of standard signed or
  //    unsigned integer type,
  return ::cuda::std::visit_format_arg(__fmt_substitute_arg_id_visitor{}, __format_arg);
}

//! These fields are a filter for which elements to parse.
//!
//! They default to false so when a new field is added it needs to be opted in
//! explicitly.
struct __fmt_spec_fields
{
  uint16_t __sign_                 : 1;
  uint16_t __alternate_form_       : 1;
  uint16_t __zero_padding_         : 1;
  uint16_t __precision_            : 1;
  uint16_t __locale_specific_form_ : 1;
  uint16_t __type_                 : 1;
  // Determines the valid values for fill.
  //
  // Originally the fill could be any character except { and }. Range-based
  // formatters use the colon to mark the beginning of the
  // underlying-format-spec. To avoid parsing ambiguities these formatter
  // specializations prohibit the use of the colon as a fill character.
  uint16_t __use_range_fill_ : 1;
  uint16_t __clear_brackets_ : 1;
  uint16_t __consume_all_    : 1;
};

// By not placing this constant in the formatter class it's not duplicated for
// char and wchar_t.
[[nodiscard]] _CCCL_API constexpr __fmt_spec_fields __fmt_spec_fields_bool() noexcept
{
  __fmt_spec_fields __ret{};
  __ret.__locale_specific_form_ = true;
  __ret.__type_                 = true;
  __ret.__consume_all_          = true;
  return __ret;
}
[[nodiscard]] _CCCL_API constexpr __fmt_spec_fields __fmt_spec_fields_int() noexcept
{
  __fmt_spec_fields __ret{};
  __ret.__sign_                 = true;
  __ret.__alternate_form_       = true;
  __ret.__zero_padding_         = true;
  __ret.__locale_specific_form_ = true;
  __ret.__type_                 = true;
  __ret.__consume_all_          = true;
  return __ret;
}
[[nodiscard]] _CCCL_API constexpr __fmt_spec_fields __fmt_spec_fields_fp() noexcept
{
  __fmt_spec_fields __ret{};
  __ret.__sign_                 = true;
  __ret.__alternate_form_       = true;
  __ret.__zero_padding_         = true;
  __ret.__locale_specific_form_ = true;
  __ret.__type_                 = true;
  __ret.__consume_all_          = true;
  return __ret;
}
[[nodiscard]] _CCCL_API constexpr __fmt_spec_fields __fmt_spec_fields_str() noexcept
{
  __fmt_spec_fields __ret{};
  __ret.__precision_   = true;
  __ret.__type_        = true;
  __ret.__consume_all_ = true;
  return __ret;
}
[[nodiscard]] _CCCL_API constexpr __fmt_spec_fields __fmt_spec_fields_ptr() noexcept
{
  __fmt_spec_fields __ret{};
  __ret.__zero_padding_ = true;
  __ret.__type_         = true;
  __ret.__consume_all_  = true;
  return __ret;
}

enum class __fmt_spec_alignment : uint8_t
{
  // No alignment is set in the format string.
  __default,
  __left,
  __center,
  __right,
  __zero_padding,
};

enum class __fmt_spec_sign : uint8_t
{
  // No sign is set in the format string.
  //
  // The sign isn't allowed for certain format-types. By using this value
  // it's possible to detect whether or not the user explicitly set the sign
  // flag. For formatting purposes it behaves the same as \ref __minus.
  __default,
  __minus,
  __plus,
  __space,
};

enum class __fmt_spec_type : uint8_t
{
  __default = 0,
  __string,
  __binary_lower_case,
  __binary_upper_case,
  __octal,
  __decimal,
  __hexadecimal_lower_case,
  __hexadecimal_upper_case,
  __pointer_lower_case,
  __pointer_upper_case,
  __char,
  __hexfloat_lower_case,
  __hexfloat_upper_case,
  __scientific_lower_case,
  __scientific_upper_case,
  __fixed_lower_case,
  __fixed_upper_case,
  __general_lower_case,
  __general_upper_case,
};

[[nodiscard]] _CCCL_API constexpr uint32_t __fmt_spec_make_type_mask(__fmt_spec_type __t)
{
  const auto __shift = static_cast<uint32_t>(__t);
  if (__shift > 31)
  {
    ::cuda::std::__throw_format_error("The type does not fit in the mask");
  }
  return 1 << __shift;
}

inline constexpr uint32_t __fmt_spec_type_mask_int =
  ::cuda::std::__fmt_spec_make_type_mask(__fmt_spec_type::__binary_lower_case) | //
  ::cuda::std::__fmt_spec_make_type_mask(__fmt_spec_type::__binary_upper_case) | //
  ::cuda::std::__fmt_spec_make_type_mask(__fmt_spec_type::__decimal) | //
  ::cuda::std::__fmt_spec_make_type_mask(__fmt_spec_type::__octal) | //
  ::cuda::std::__fmt_spec_make_type_mask(__fmt_spec_type::__hexadecimal_lower_case) | //
  ::cuda::std::__fmt_spec_make_type_mask(__fmt_spec_type::__hexadecimal_upper_case);

// there is a bug in gcc < 10 that warns about `__fmt_spec_alignment : 3` being too small for all enum values,
// so we use the underlying type instead
struct __fmt_spec_std
{
  underlying_type_t<__fmt_spec_alignment> __alignment_ : 3;
  underlying_type_t<__fmt_spec_sign> __sign_           : 2;
  bool __alternate_form_                               : 1;
  bool __locale_specific_form_                         : 1;
  bool __padding_0_                                    : 1;
  __fmt_spec_type __type_;
};

struct __fmt_spec_chrono
{
  underlying_type_t<__fmt_spec_alignment> __alignment_ : 3;
  bool __locale_specific_form_                         : 1;
  bool __hour_                                         : 1;
  bool __weekday_name_                                 : 1;
  bool __weekday_                                      : 1;
  bool __day_of_year_                                  : 1;
  bool __week_of_year_                                 : 1;
  bool __month_name_                                   : 1;
};

//! The fill UCS scalar value.
//!
//! This is always an array, with 1, 2, or 4 elements.
//! The size of the data structure is always 32-bits.
template <class _CharT>
struct __fmt_spec_code_point;

template <>
struct __fmt_spec_code_point<char>
{
  char __data[4] = {' '};
};

#if _CCCL_HAS_WCHAR_T()
template <>
struct __fmt_spec_code_point<wchar_t>
{
  wchar_t __data[4 / sizeof(wchar_t)] = {L' '};
};
#endif // _CCCL_HAS_WCHAR_T()

//! Contains the parsed formatting specifications.
//!
//! This contains information for both the std-format-spec and the
//! chrono-format-spec. This results in some unused members for both
//! specifications. However these unused members don't increase the size
//! of the structure.
//!
//! This struct doesn't cross ABI boundaries so its layout doesn't need to be
//! kept stable.
template <class _CharT>
struct __fmt_parsed_spec
{
  union
  {
    // The field __alignment_ is the first element in __std_ and __chrono_.
    // This allows the code to always inspect this value regardless of which member
    // of the union is the active member [class.union.general]/2.
    //
    // This is needed since the generic output routines handle the alignment of
    // the output.
    underlying_type_t<__fmt_spec_alignment> __alignment_ : 3;
    __fmt_spec_std __std_;
    __fmt_spec_chrono __chrono_;
  };

  //! The requested width.
  //!
  //! When the format-spec used an arg-id for this field it has already been
  //! replaced with the value of that arg-id.
  uint32_t __width_;

  //! The requested precision.
  //!
  //! When the format-spec used an arg-id for this field it has already been
  //! replaced with the value of that arg-id.
  int32_t __precision_;

  __fmt_spec_code_point<_CharT> __fill_;

  [[nodiscard]] _CCCL_API constexpr bool __has_width() const
  {
    return __width_ > 0;
  }

  [[nodiscard]] _CCCL_API constexpr bool __has_precision() const
  {
    return __precision_ >= 0;
  }
};

// Validate the struct is small and cheap to copy since the struct is passed by
// value in formatting functions.
static_assert(sizeof(__fmt_parsed_spec<char>) == 16);
static_assert(is_trivially_copyable_v<__fmt_parsed_spec<char>>);
#if _CCCL_HAS_WCHAR_T()
static_assert(sizeof(__fmt_parsed_spec<wchar_t>) == 16);
static_assert(is_trivially_copyable_v<__fmt_parsed_spec<wchar_t>>);
#endif // _CCCL_HAS_WCHAR_T()

//! The parser for the std-format-spec.
//!
//! Note this class is a member of std::formatter specializations. It's
//! expected developers will create their own formatter specializations that
//! inherit from the std::formatter specializations. This means this class
//! must be ABI stable. To aid the stability the unused bits in the class are
//! set to zero. That way they can be repurposed if a future revision of the
//! Standards adds new fields to std-format-spec.
template <class _CharT>
class __fmt_spec_parser
{
public:
  //! Parses the format specification.
  //!
  //! Depending on whether the parsing is done compile-time or run-time
  //! the method slightly differs.
  //! - Only parses a field when it is in the __fields. Accepting all
  //!   fields and then validating the valid ones has a performance impact.
  //!   This is faster but gives slightly worse error messages.
  //! - At compile-time when a field is not accepted the parser will still
  //!   parse it and give an error when it's present. This gives a more
  //!   accurate error.
  //! The idea is that most times the format instead of the vformat
  //! functions are used. In that case the error will be detected during
  //! compilation and there is no need to pay for the run-time overhead.
  template <class _ParseContext>
  _CCCL_API constexpr typename _ParseContext::iterator __parse(_ParseContext& __ctx, __fmt_spec_fields __fields)
  {
    auto __begin = __ctx.begin();
    auto __end   = __ctx.end();
    if (__begin == __end || *__begin == _CharT{'}'} || (__fields.__use_range_fill_ && *__begin == _CharT{':'}))
    {
      return __begin;
    }

    if (__parse_fill_align(__begin, __end) && __begin == __end)
    {
      return __begin;
    }

    if (__fields.__sign_)
    {
      if (__parse_sign(__begin) && __begin == __end)
      {
        return __begin;
      }
    }
    else
    {
      _CCCL_IF_CONSTEVAL_DEFAULT
      {
        if (__parse_sign(__begin))
        {
          ::cuda::std::__throw_format_error("The format specification does not allow the sign option");
        }
      }
    }

    if (__fields.__alternate_form_)
    {
      if (__parse_alternate_form(__begin) && __begin == __end)
      {
        return __begin;
      }
    }
    else
    {
      _CCCL_IF_CONSTEVAL_DEFAULT
      {
        if (__parse_alternate_form(__begin))
        {
          ::cuda::std::__throw_format_error("The format specifier does not allow the alternate form option");
        }
      }
    }

    if (__fields.__zero_padding_)
    {
      if (__parse_zero_padding(__begin) && __begin == __end)
      {
        return __begin;
      }
    }
    else
    {
      _CCCL_IF_CONSTEVAL_DEFAULT
      {
        if (__parse_zero_padding(__begin))
        {
          ::cuda::std::__throw_format_error("The format specifier does not allow the zero-padding option");
        }
      }
    }

    if (__parse_width(__begin, __end, __ctx) && __begin == __end)
    {
      return __begin;
    }

    if (__fields.__precision_)
    {
      if (__parse_precision(__begin, __end, __ctx) && __begin == __end)
      {
        return __begin;
      }
    }
    else
    {
      _CCCL_IF_CONSTEVAL_DEFAULT
      {
        if (__parse_precision(__begin, __end, __ctx))
        {
          ::cuda::std::__throw_format_error("The format specifier does not allow the precision option");
        }
      }
    }

    if (__fields.__locale_specific_form_)
    {
      if (__parse_locale_specific_form(__begin) && __begin == __end)
      {
        return __begin;
      }
    }
    else
    {
      _CCCL_IF_CONSTEVAL_DEFAULT
      {
        if (__parse_locale_specific_form(__begin))
        {
          ::cuda::std::__throw_format_error("The format specifier does not allow the locale-specific form option");
        }
      }
    }

    if (__fields.__clear_brackets_)
    {
      if (__parse_clear_brackets(__begin) && __begin == __end)
      {
        return __begin;
      }
    }
    else
    {
      _CCCL_IF_CONSTEVAL_DEFAULT
      {
        if (__parse_clear_brackets(__begin))
        {
          ::cuda::std::__throw_format_error("The format specifier does not allow the n option");
        }
      }
    }

    if (__fields.__type_)
    {
      __parse_type(__begin);
    }

    if (!__fields.__consume_all_)
    {
      return __begin;
    }

    if (__begin != __end && *__begin != _CharT{'}'})
    {
      ::cuda::std::__throw_format_error("The format specifier should consume the input or end with a '}'");
    }

    return __begin;
  }

  //! Validates the selected the parsed data.
  //!
  //! The valid fields in the parser may depend on the display type
  //! selected. But the type is the last optional field, so by the time
  //! it's known an option can't be used, it already has been parsed.
  //! This does the validation again.
  //!
  //! For example an integral may have a sign, zero-padding, or alternate
  //! form when the type option is not 'c'. So the generic approach is:
  //!
  //! typename _ParseContext::iterator __result = __parser_.__parse(__ctx, __format_spec::__fields_integral);
  //! if (__parser.__type_ == __format_spec::__type::__char) {
  //!   __parser.__validate((__format_spec::__fields_bool, "an integer");
  //!   ... // more char adjustments
  //! } else {
  //!   ... // validate an integral type.
  //! }
  //!
  //! For some types all valid options need a second validation run, like
  //! boolean types.
  //!
  //! Depending on whether the validation is done at compile-time or
  //! run-time the error differs
  //! - run-time the exception is thrown and contains the type of field
  //!   being validated.
  //! - at compile-time the line with `std::__throw_format_error` is shown
  //!   in the output. In that case it's important for the error to be on one
  //!   line.
  //! Note future versions of C++ may allow better compile-time error
  //! reporting.
  template <size_t _IdSize>
  _CCCL_API constexpr void
  __validate(__fmt_spec_fields __fields, const char (&__id)[_IdSize], uint32_t __type_mask = ~uint32_t{0}) const
  {
    if (!__fields.__sign_ && __fmt_spec_sign{__sign_} != __fmt_spec_sign::__default)
    {
      _CCCL_IF_CONSTEVAL
      {
        ::cuda::std::__throw_format_error("The format specifier does not allow the sign option");
      }
      else
      {
        ::cuda::std::__throw_invalid_option_format_error(__id, "sign");
      }
    }

    if (!__fields.__alternate_form_ && __alternate_form_)
    {
      _CCCL_IF_CONSTEVAL
      {
        ::cuda::std::__throw_format_error("The format specifier does not allow the alternate form option");
      }
      else
      {
        ::cuda::std::__throw_invalid_option_format_error(__id, "alternate form");
      }
    }

    if (!__fields.__zero_padding_ && __fmt_spec_alignment{__alignment_} == __fmt_spec_alignment::__zero_padding)
    {
      _CCCL_IF_CONSTEVAL
      {
        ::cuda::std::__throw_format_error("The format specifier does not allow the zero-padding option");
      }
      else
      {
        ::cuda::std::__throw_invalid_option_format_error(__id, "zero-padding");
      }
    }

    if (!__fields.__precision_ && __precision_ != -1)
    { // Works both when the precision has a value or an arg-id.
      _CCCL_IF_CONSTEVAL
      {
        ::cuda::std::__throw_format_error("The format specifier does not allow the precision option");
      }
      else
      {
        ::cuda::std::__throw_invalid_option_format_error(__id, "precision");
      }
    }

    if (!__fields.__locale_specific_form_ && __locale_specific_form_)
    {
      _CCCL_IF_CONSTEVAL
      {
        ::cuda::std::__throw_format_error("The format specifier does not allow the locale-specific form option");
      }
      else
      {
        ::cuda::std::__throw_invalid_option_format_error(__id, "locale-specific form");
      }
    }

    if ((::cuda::std::__fmt_spec_make_type_mask(__type_) & __type_mask) == 0)
    {
      _CCCL_IF_CONSTEVAL
      {
        ::cuda::std::__throw_format_error("The format specifier uses an invalid value for the type option");
      }
      else
      {
        ::cuda::std::__throw_invalid_type_format_error(__id);
      }
    }
  }

  //! Returns the `__fmt_parsed_spec` with the resolved dynamic sizes.
  template <class _Ctx>
  [[nodiscard]] _CCCL_API __fmt_parsed_spec<_CharT> __get_parsed_std_spec(_Ctx& __ctx) const
  {
    __fmt_parsed_spec<_CharT> __ret{};
    __ret.__std_.__alignment_            = __alignment_;
    __ret.__std_.__sign_                 = __sign_;
    __ret.__std_.__alternate_form_       = __alternate_form_;
    __ret.__std_.__locale_specific_form_ = __locale_specific_form_;
    __ret.__std_.__type_                 = __type_;
    __ret.__width_                       = __get_width(__ctx);
    __ret.__precision_                   = __get_precision(__ctx);
    __ret.__fill_                        = __fill_;
    return __ret;
  }

  //! Returns the `__fmt_parsed_spec` with the resolved dynamic sizes.
  template <class _Ctx>
  [[nodiscard]] _CCCL_API __fmt_parsed_spec<_CharT> __get_parsed_chrono_spec(_Ctx& __ctx) const
  {
    __fmt_parsed_spec<_CharT> __ret{};
    __ret.__chrono_.__alignment_            = __alignment_;
    __ret.__chrono_.__locale_specific_form_ = __locale_specific_form_;
    __ret.__chrono_.__hour_                 = __hour_;
    __ret.__chrono_.__weekday_name_         = __weekday_name_;
    __ret.__chrono_.__weekday_              = __weekday_;
    __ret.__chrono_.__day_of_year_          = __day_of_year_;
    __ret.__chrono_.__week_of_year_         = __week_of_year_;
    __ret.__chrono_.__month_name_           = __month_name_;
    __ret.__width_                          = __get_width(__ctx);
    __ret.__precision_                      = __get_precision(__ctx);
    __ret.__fill_                           = __fill_;
    return __ret;
  }

  _CCCL_API constexpr __fmt_spec_parser() noexcept
      : __alignment_{::cuda::std::to_underlying(__fmt_spec_alignment::__default)}
      , __sign_{::cuda::std::to_underlying(__fmt_spec_sign::__default)}
      , __alternate_form_{false}
      , __locale_specific_form_{false}
      , __clear_brackets_{false}
      , __type_{__fmt_spec_type::__default}
      , __hour_{false}
      , __weekday_name_{false}
      , __weekday_{false}
      , __day_of_year_{false}
      , __week_of_year_{false}
      , __month_name_{false}
      , __reserved_0_{}
      , __reserved_1_{}
      , __width_as_arg_{false}
      , __precision_as_arg_{false}
      , __width_{0}
      , __precision_{-1}
      , __fill_{}
  {}

  underlying_type_t<__fmt_spec_alignment> __alignment_ : 3;
  underlying_type_t<__fmt_spec_sign> __sign_           : 2;
  bool __alternate_form_                               : 1;
  bool __locale_specific_form_                         : 1;
  bool __clear_brackets_                               : 1;
  __fmt_spec_type __type_;

  // These flags are only used for formatting chrono. Since the struct has
  // padding space left it's added to this structure.
  bool __hour_ : 1;

  bool __weekday_name_ : 1;
  bool __weekday_      : 1;

  bool __day_of_year_  : 1;
  bool __week_of_year_ : 1;

  bool __month_name_ : 1;

  uint8_t __reserved_0_ : 2;
  uint8_t __reserved_1_ : 6;
  // These two flags are only used internally and not part of the
  // __parsed_specifications. Therefore put them at the end.
  bool __width_as_arg_     : 1;
  bool __precision_as_arg_ : 1;

  // The requested width, either the value or the arg-id.
  uint32_t __width_;

  // The requested precision, either the value or the arg-id.
  int32_t __precision_;

  __fmt_spec_code_point<_CharT> __fill_;

private:
  template <class _It, class _ParseCtx>
  [[nodiscard]] _CCCL_API static constexpr __fmt_parse_number_result<_It>
  __parse_arg_id(_It __begin, _It __end, _ParseCtx& __ctx)
  {
    // This function is a wrapper to call the real parser. But it does the
    // validation for the pre-conditions and post-conditions.
    if (__begin == __end)
    {
      ::cuda::std::__throw_format_error("End of input while parsing an argument index");
    }

    auto __r = ::cuda::std::__fmt_parse_arg_id(__begin, __end, __ctx);

    if (__r.__last == __end || *__r.__last != _CharT{'}'})
    {
      ::cuda::std::__throw_format_error("The argument index is invalid");
    }

    ++__r.__last;
    return __r;
  }

  [[nodiscard]] _CCCL_API constexpr bool __parse_alignment(_CharT __c)
  {
    switch (__c)
    {
      case _CharT{'<'}:
        __alignment_ = ::cuda::std::to_underlying(__fmt_spec_alignment::__left);
        return true;

      case _CharT{'^'}:
        __alignment_ = ::cuda::std::to_underlying(__fmt_spec_alignment::__center);
        return true;

      case _CharT{'>'}:
        __alignment_ = ::cuda::std::to_underlying(__fmt_spec_alignment::__right);
        return true;
    }
    return false;
  }

  _CCCL_API constexpr void __validate_fill_character(_CharT __fill)
  {
    // The forbidden fill characters all code points formed from a single code unit, thus the
    // check can be omitted when more code units are used.
    if (__fill == _CharT{'{'})
    {
      ::cuda::std::__throw_format_error("The fill option contains an invalid value");
    }
  }

  // range-fill and tuple-fill are identical
  template <class _It>
  [[nodiscard]] _CCCL_API constexpr bool __parse_fill_align(_It& __begin, _It __end)
  {
    _CCCL_ASSERT(__begin != __end,
                 "when called with an empty input the function will cause "
                 "undefined behavior by evaluating data not in the input");
    if (__begin + 1 != __end)
    {
      if (__parse_alignment(*(__begin + 1)))
      {
        __validate_fill_character(*__begin);

        __fill_.__data[0] = *__begin;
        __begin += 2;
        return true;
      }
    }

    if (!__parse_alignment(*__begin))
    {
      return false;
    }

    ++__begin;
    return true;
  }

  template <class _It>
  [[nodiscard]] _CCCL_API constexpr bool __parse_sign(_It& __begin)
  {
    switch (*__begin)
    {
      case _CharT{'-'}:
        __sign_ = ::cuda::std::to_underlying(__fmt_spec_sign::__minus);
        break;
      case _CharT{'+'}:
        __sign_ = ::cuda::std::to_underlying(__fmt_spec_sign::__plus);
        break;
      case _CharT{' '}:
        __sign_ = ::cuda::std::to_underlying(__fmt_spec_sign::__space);
        break;
      default:
        return false;
    }
    ++__begin;
    return true;
  }

  template <class _It>
  [[nodiscard]] _CCCL_API constexpr bool __parse_alternate_form(_It& __begin)
  {
    if (*__begin != _CharT{'#'})
    {
      return false;
    }
    __alternate_form_ = true;
    ++__begin;
    return true;
  }

  template <class _It>
  [[nodiscard]] _CCCL_API constexpr bool __parse_zero_padding(_It& __begin)
  {
    if (*__begin != _CharT{'0'})
    {
      return false;
    }
    if (__fmt_spec_alignment{__alignment_} == __fmt_spec_alignment::__default)
    {
      __alignment_ = ::cuda::std::to_underlying(__fmt_spec_alignment::__zero_padding);
    }
    ++__begin;
    return true;
  }

  template <class _It, class _Ctx>
  [[nodiscard]] _CCCL_API constexpr bool __parse_width(_It& __begin, _It __end, _Ctx& __ctx)
  {
    if (*__begin == _CharT{'0'})
    {
      ::cuda::std::__throw_format_error("The width option should not have a leading zero");
    }

    if (*__begin == _CharT{'{'})
    {
      const auto __r  = __parse_arg_id(++__begin, __end, __ctx);
      __width_as_arg_ = true;
      __width_        = __r.__value;
      __begin         = __r.__last;
      return true;
    }

    if (*__begin < _CharT{'0'} || *__begin > _CharT{'9'})
    {
      return false;
    }

    const auto __r = ::cuda::std::__fmt_parse_number(__begin, __end);
    __width_       = __r.__value;
    _CCCL_ASSERT(__width_ != 0,
                 "A zero value isn't allowed and should be impossible, due to validations in this function");
    __begin = __r.__last;
    return true;
  }

  template <class _It, class _Ctx>
  [[nodiscard]] _CCCL_API constexpr bool __parse_precision(_It& __begin, _It __end, _Ctx& __ctx)
  {
    if (*__begin != _CharT{'.'})
    {
      return false;
    }

    ++__begin;
    if (__begin == __end)
    {
      ::cuda::std::__throw_format_error("End of input while parsing format specifier precision");
    }

    if (*__begin == _CharT{'{'})
    {
      const auto __arg_id = __parse_arg_id(++__begin, __end, __ctx);
      __precision_as_arg_ = true;
      __precision_        = __arg_id.__value;
      __begin             = __arg_id.__last;
      return true;
    }

    if (*__begin < _CharT{'0'} || *__begin > _CharT{'9'})
    {
      ::cuda::std::__throw_format_error("The precision option does not contain a value or an argument index");
    }

    const auto __r      = ::cuda::std::__fmt_parse_number(__begin, __end);
    __precision_        = __r.__value;
    __precision_as_arg_ = false;
    __begin             = __r.__last;
    return true;
  }

  template <class _It>
  [[nodiscard]] _CCCL_API constexpr bool __parse_locale_specific_form(_It& __begin)
  {
    if (*__begin != _CharT{'L'})
    {
      return false;
    }
    __locale_specific_form_ = true;
    ++__begin;
    return true;
  }

  template <class _It>
  [[nodiscard]] _CCCL_API constexpr bool __parse_clear_brackets(_It& __begin)
  {
    if (*__begin != _CharT{'n'})
    {
      return false;
    }
    __clear_brackets_ = true;
    ++__begin;
    return true;
  }

  template <class _It>
  _CCCL_API constexpr void __parse_type(_It& __begin)
  {
    // Determines the type. It does not validate whether the selected type is
    // valid. Most formatters have optional fields that are only allowed for
    // certain types. These parsers need to do validation after the type has
    // been parsed. So its easier to implement the validation for all types in
    // the specific parse function.
    switch (*__begin)
    {
      case 'A':
        __type_ = __fmt_spec_type::__hexfloat_upper_case;
        break;
      case 'B':
        __type_ = __fmt_spec_type::__binary_upper_case;
        break;
      case 'E':
        __type_ = __fmt_spec_type::__scientific_upper_case;
        break;
      case 'F':
        __type_ = __fmt_spec_type::__fixed_upper_case;
        break;
      case 'G':
        __type_ = __fmt_spec_type::__general_upper_case;
        break;
      case 'X':
        __type_ = __fmt_spec_type::__hexadecimal_upper_case;
        break;
      case 'a':
        __type_ = __fmt_spec_type::__hexfloat_lower_case;
        break;
      case 'b':
        __type_ = __fmt_spec_type::__binary_lower_case;
        break;
      case 'c':
        __type_ = __fmt_spec_type::__char;
        break;
      case 'd':
        __type_ = __fmt_spec_type::__decimal;
        break;
      case 'e':
        __type_ = __fmt_spec_type::__scientific_lower_case;
        break;
      case 'f':
        __type_ = __fmt_spec_type::__fixed_lower_case;
        break;
      case 'g':
        __type_ = __fmt_spec_type::__general_lower_case;
        break;
      case 'o':
        __type_ = __fmt_spec_type::__octal;
        break;
      case 'p':
        __type_ = __fmt_spec_type::__pointer_lower_case;
        break;
      case 'P':
        __type_ = __fmt_spec_type::__pointer_upper_case;
        break;
      case 's':
        __type_ = __fmt_spec_type::__string;
        break;
      case 'x':
        __type_ = __fmt_spec_type::__hexadecimal_lower_case;
        break;
      default:
        return;
    }
    ++__begin;
  }

  template <class _Ctx>
  [[nodiscard]] _CCCL_API uint32_t __get_width(_Ctx& __ctx) const
  {
    if (!__width_as_arg_)
    {
      return __width_;
    }
    return ::cuda::std::__fmt_substitute_arg_id(__ctx.arg(__width_));
  }

  template <class _Ctx>
  [[nodiscard]] _CCCL_API int32_t __get_precision(_Ctx& __ctx) const
  {
    if (!__precision_as_arg_)
    {
      return __precision_;
    }
    return ::cuda::std::__fmt_substitute_arg_id(__ctx.arg(__precision_));
  }
};

// Validates whether the reserved bitfields don't change the size.
static_assert(sizeof(__fmt_spec_parser<char>) == 16);
#if _CCCL_HAS_WCHAR_T()
static_assert(sizeof(__fmt_spec_parser<wchar_t>) == 16);
#endif // _CCCL_HAS_WCHAR_T()

_CCCL_API constexpr void __fmt_process_display_type_str(__fmt_spec_type __type)
{
  switch (__type)
  {
    case __fmt_spec_type::__default:
    case __fmt_spec_type::__string:
      break;
    default:
      ::cuda::std::__throw_format_error("The type option contains an invalid value for a string formatting argument");
  }
}

template <class _CharT, size_t _IdSize>
_CCCL_API constexpr void
__fmt_process_display_type_bool_str(__fmt_spec_parser<_CharT>& __parser, const char (&__id)[_IdSize])
{
  __parser.__validate(::cuda::std::__fmt_spec_fields_bool(), __id);
  if (__fmt_spec_alignment{__parser.__alignment_} == __fmt_spec_alignment::__default)
  {
    __parser.__alignment_ = ::cuda::std::to_underlying(__fmt_spec_alignment::__left);
  }
}

template <class _CharT, size_t _IdSize>
_CCCL_API constexpr void
__fmt_process_display_type_char(__fmt_spec_parser<_CharT>& __parser, const char (&__id)[_IdSize])
{
  ::cuda::std::__fmt_process_display_type_bool_str(__parser, __id);
}

template <class _CharT>
_CCCL_API constexpr void __fmt_process_parsed_bool(__fmt_spec_parser<_CharT>& __parser)
{
  constexpr auto& __id = "a bool";

  switch (__parser.__type_)
  {
    case __fmt_spec_type::__default:
    case __fmt_spec_type::__string:
      ::cuda::std::__fmt_process_display_type_bool_str(__parser, __id);
      break;
    case __fmt_spec_type::__binary_lower_case:
    case __fmt_spec_type::__binary_upper_case:
    case __fmt_spec_type::__octal:
    case __fmt_spec_type::__decimal:
    case __fmt_spec_type::__hexadecimal_lower_case:
    case __fmt_spec_type::__hexadecimal_upper_case:
      break;
    default:
      ::cuda::std::__throw_invalid_type_format_error(__id);
  }
}

template <class _CharT>
_CCCL_API constexpr void __fmt_process_parsed_char(__fmt_spec_parser<_CharT>& __parser)
{
  constexpr auto& __id = "a character";

  switch (__parser.__type_)
  {
    case __fmt_spec_type::__default:
    case __fmt_spec_type::__char:
      ::cuda::std::__fmt_process_display_type_char(__parser, __id);
      break;
    case __fmt_spec_type::__binary_lower_case:
    case __fmt_spec_type::__binary_upper_case:
    case __fmt_spec_type::__octal:
    case __fmt_spec_type::__decimal:
    case __fmt_spec_type::__hexadecimal_lower_case:
    case __fmt_spec_type::__hexadecimal_upper_case:
      break;
    default:
      ::cuda::std::__throw_invalid_type_format_error(__id);
  }
}

template <class _CharT>
_CCCL_API constexpr void __fmt_process_parsed_int(__fmt_spec_parser<_CharT>& __parser)
{
  constexpr auto& __id = "an integer";

  switch (__parser.__type_)
  {
    case __fmt_spec_type::__default:
    case __fmt_spec_type::__binary_lower_case:
    case __fmt_spec_type::__binary_upper_case:
    case __fmt_spec_type::__octal:
    case __fmt_spec_type::__decimal:
    case __fmt_spec_type::__hexadecimal_lower_case:
    case __fmt_spec_type::__hexadecimal_upper_case:
      break;
    case __fmt_spec_type::__char:
      ::cuda::std::__fmt_process_display_type_char(__parser, __id);
      break;
    default:
      ::cuda::std::__throw_invalid_type_format_error(__id);
  }
}

template <class _CharT>
_CCCL_API constexpr void __fmt_process_parsed_fp(__fmt_spec_parser<_CharT>& __parser)
{
  switch (__parser.__type_)
  {
    case __fmt_spec_type::__default:
    case __fmt_spec_type::__hexfloat_lower_case:
    case __fmt_spec_type::__hexfloat_upper_case:
      // Precision specific behavior will be handled later.
      break;
    case __fmt_spec_type::__scientific_lower_case:
    case __fmt_spec_type::__scientific_upper_case:
    case __fmt_spec_type::__fixed_lower_case:
    case __fmt_spec_type::__fixed_upper_case:
    case __fmt_spec_type::__general_lower_case:
    case __fmt_spec_type::__general_upper_case:
      if (!__parser.__precision_as_arg_ && __parser.__precision_ == -1)
      {
        // Set the default precision for the call to to_chars.
        __parser.__precision_ = 6;
      }
      break;
    default:
      ::cuda::std::__throw_invalid_type_format_error("a floating-point");
  }
}

_CCCL_API constexpr void __fmt_process_display_type_ptr(__fmt_spec_type __type)
{
  switch (__type)
  {
    case __fmt_spec_type::__default:
    case __fmt_spec_type::__pointer_lower_case:
    case __fmt_spec_type::__pointer_upper_case:
      break;
    default:
      ::cuda::std::__throw_invalid_type_format_error("a pointer");
  }
}

template <class _It>
struct __fmt_column_width_result
{
  // The number of output columns.
  size_t __width_;
  // One beyond the last code unit used in the estimation.
  //
  // This limits the original output to fit in the wanted number of columns.
  _It __last_;
};

template <class _It>
_CCCL_HOST_DEVICE __fmt_column_width_result(size_t, _It) -> __fmt_column_width_result<_It>;

//! Since a column width can be two it's possible that the requested column
//! width can't be achieved. Depending on the intended usage the policy can be
//! selected.
//! - When used as precision the maximum width may not be exceeded and the
//!   result should be "rounded down" to the previous boundary.
//! - When used as a width we're done once the minimum is reached, but
//!   exceeding is not an issue. Rounding down is an issue since that will
//!   result in writing fill characters. Therefore the result needs to be
//!   "rounded up".
enum class __fmt_column_width_rounding
{
  __down,
  __up,
};

template <class _CharT>
[[nodiscard]] _CCCL_API constexpr __fmt_column_width_result<typename basic_string_view<_CharT>::const_iterator>
__fmt_estimate_column_width(basic_string_view<_CharT> __str, size_t __maximum, __fmt_column_width_rounding) noexcept
{
  // When Unicode isn't supported assume ASCII and every code unit is one code
  // point. In ASCII the estimated column width is always one. Thus there's no
  // need for rounding.
  size_t __width = ::cuda::std::min(__str.size(), __maximum);
  return {__width, __str.begin() + __width};
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD__FORMAT_FORMAT_SPEC_PARSER_H
