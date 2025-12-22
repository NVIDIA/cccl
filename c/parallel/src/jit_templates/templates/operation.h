//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
#  include <cuda/std/cstddef>
#  include <cuda/std/type_traits>
#  include <cuda/std/utility>

#  include <cccl/c/types.h>
#  include <util/types.h>
#endif

#include "../mappings/operation.h"
#include "../mappings/type_info.h"

template <cccl_type_info_mapping RetT>
using cccl_mapping_to_type = decltype(RetT)::Type;

template <typename Tag, cccl_op_t_mapping Operation, cccl_type_info_mapping RetT, cccl_type_info_mapping... ArgTs>
struct stateless_user_operation
{
  // Note: The user provided C f  unction (Operation.operation) must match the signature:
  // void (void* arg1, ..., void* argN, void* result_ptr)
  __device__ cccl_mapping_to_type<RetT> operator()(cccl_mapping_to_type<ArgTs>... args) const
  {
    using TargetCFuncPtr = void (*)(const decltype(args, void())*..., void*);

    // Cast the stored operation pointer (assumed to be void* or compatible)
    auto c_func_ptr = reinterpret_cast<TargetCFuncPtr>(Operation.operation);

    // Prepare storage for the result
    typename decltype(RetT)::Type result;

    // Call the C function, casting argument addresses to void*
    c_func_ptr((const_cast<void*>(static_cast<const void*>(&args)))..., &result);

    return result;
  }
};

template <cuda::std::size_t Size, cuda::std::size_t Alignment>
struct alignas(Alignment) user_operation_state
{
  char data[Size];
};

template <typename Tag, cccl_op_t_mapping Operation, cccl_type_info_mapping RetT, cccl_type_info_mapping... ArgTs>
struct stateful_user_operation
{
  user_operation_state<Operation.size, Operation.alignment> state;
  __device__ cccl_mapping_to_type<RetT> operator()(cccl_mapping_to_type<ArgTs>... args)
  {
    // Note: The user provided C function (Operation.operation) must match the signature:
    // void (void* state, void* arg1, ..., void* argN, void* result_ptr)
    using TargetCFuncPtr = void (*)(void*, const decltype(args, void())*..., void*);

    // Cast the stored operation pointer (assumed to be void* or compatible)
    auto c_func_ptr = reinterpret_cast<TargetCFuncPtr>(Operation.operation);

    // Prepare storage for the result
    typename decltype(RetT)::Type result;

    // Call the C function, passing state address, casting argument addresses to void*, and result pointer
    c_func_ptr(&state, (const_cast<void*>(static_cast<const void*>(&args)))..., &result);

    return result;
  }
};

struct user_operation_traits
{
  static const constexpr auto name = "user_operation_traits::type";
  template <typename Tag, cccl_op_t_mapping Operation, cccl_type_info_mapping RetT, cccl_type_info_mapping... ArgTs>
  using type = cuda::std::conditional_t<Operation.is_stateless,
                                        stateless_user_operation<Tag, Operation, RetT, ArgTs...>,
                                        stateful_user_operation<Tag, Operation, RetT, ArgTs...>>;

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
  using operator_builder = std::string (*)(cuda::std::span<std::string_view>, const char*, const char*);
  using argument_matcher = operator_builder (*)(cuda::std::span<cccl_type_enum>, std::string_view);

  static std::string unary_builder(cuda::std::span<std::string_view> types, const char* symbol, const char* name_)
  {
    return std::format(
      R"(
extern "C" __device__ void {0}(const void*, void*);
__device__ {1} operator{2}(const {3} & arg)
{{
  {1} ret;
  {0}(&arg, &ret);
  return ret;
}}
                       )",
      name_, // 0
      types[0], // 1
      symbol, // 2
      types[1] // 3
    );
  }

  static std::string binary_builder(cuda::std::span<std::string_view> types, const char* symbol, const char* name_)
  {
    return std::format(
      R"(
extern "C" __device__ void {0}(const void*, const void*, void*);
__device__ {1} operator{2}(const {3} & lhs, const {3} & rhs)
{{
    {1} ret;
    {0}(&lhs, &rhs, &ret);
    return ret;
}}
                       )",
      name_, // 0
      types[0], // 1
      symbol, // 2
      types[1] // 3
    );
  }

  static operator_builder unary_matcher(cuda::std::span<cccl_type_enum> types, std::string_view name_)
  {
    if (types.size() != 2)
    {
      throw std::runtime_error(std::format(
        "c.parallel: well-known operation '{}' expected 1 argument, received {}.", name_, types.size() - 1));
    }
    if (types[0] != types[1])
    {
      throw std::runtime_error(std::format(
        "c.parallel: well-known operation '{}' expected to return its argument type ({}), but returns {}.",
        name_,
        cccl_type_enum_to_name(types[1]),
        cccl_type_enum_to_name(types[0])));
    }

    return unary_builder;
  }

  static operator_builder binary_matcher(cuda::std::span<cccl_type_enum> types, std::string_view name_)
  {
    if (types.size() != 3)
    {
      throw std::runtime_error(std::format(
        "c.parallel: well-known operation '{}' expected 2 arguments, received {}.", name_, types.size() - 1));
    }
    if (types[1] != types[2])
    {
      throw std::runtime_error(std::format(
        "c.parallel: well-known operation '{}' expected to have matching argument types, but has argument types {} and "
        "{}.",
        name_,
        cccl_type_enum_to_name(types[1]),
        cccl_type_enum_to_name(types[2])));
    }
    if (types[0] != types[1])
    {
      throw std::runtime_error(std::format(
        "c.parallel: well-known operation '{}' expected to return its argument type ({}), but returns {}.",
        name_,
        cccl_type_enum_to_name(types[1]),
        cccl_type_enum_to_name(types[0])));
    }

    return binary_builder;
  }

  static operator_builder unary_predicate_matcher(cuda::std::span<cccl_type_enum> types, std::string_view name_)
  {
    if (types.size() != 2)
    {
      throw std::runtime_error(std::format(
        "c.parallel: well-known operation '{}' expected 1 argument, received {}.", name_, types.size() - 1));
    }
    if (types[0] != cccl_type_enum::CCCL_BOOLEAN)
    {
      throw std::runtime_error(
        std::format("c.parallel: well-known operation '{}' expected to return boolean, but returns {}.",
                    name_,
                    cccl_type_enum_to_name(types[0])));
    }

    return unary_builder;
  }

  static operator_builder binary_predicate_matcher(cuda::std::span<cccl_type_enum> types, std::string_view name_)
  {
    if (types.size() != 3)
    {
      throw std::runtime_error(std::format(
        "c.parallel: well-known operation '{}' expected 2 arguments, received {}.", name_, types.size() - 1));
    }
    if (types[1] != types[2])
    {
      throw std::runtime_error(std::format(
        "c.parallel: well-known operation '{}' expected to have matching argument types, but has argument types {} and "
        "{}.",
        name_,
        cccl_type_enum_to_name(types[1]),
        cccl_type_enum_to_name(types[2])));
    }
    if (types[0] != cccl_type_enum::CCCL_BOOLEAN)
    {
      throw std::runtime_error(
        std::format("c.parallel: well-known operation '{}' expected to return boolean, but returns {}.",
                    name_,
                    cccl_type_enum_to_name(types[0])));
    }

    return binary_builder;
  }

  struct well_known_description
  {
    std::format_string<const char*> name;
    argument_matcher check;
    std::optional<const char*> symbol = std::nullopt;

    constexpr well_known_description(
      std::format_string<const char*> name, argument_matcher check, std::optional<const char*> symbol = std::nullopt)
        : name(name)
        , check(check)
        , symbol(symbol)
    {}
  };

  static cuda::std::optional<well_known_description> well_known_operation_description(cccl_op_kind_t kind)
  {
    switch (kind)
    {
      case cccl_op_kind_t::CCCL_STATELESS:
        return cuda::std::nullopt;
      case cccl_op_kind_t::CCCL_STATEFUL:
        return cuda::std::nullopt;
      case cccl_op_kind_t::CCCL_PLUS:
        return well_known_description{"::cuda::std::plus<{}>", binary_matcher, "+"};
      case cccl_op_kind_t::CCCL_MINUS:
        return well_known_description{"::cuda::std::minus<{}>", binary_matcher, "-"};
      case cccl_op_kind_t::CCCL_MULTIPLIES:
        return well_known_description{"::cuda::std::multiplies<{}>", binary_matcher, "*"};
      case cccl_op_kind_t::CCCL_DIVIDES:
        return well_known_description{"::cuda::std::divides<{}>", binary_matcher, "/"};
      case cccl_op_kind_t::CCCL_MODULUS:
        return well_known_description{"::cuda::std::modulus<{}>", binary_matcher, "%"};
      case cccl_op_kind_t::CCCL_EQUAL_TO:
        return well_known_description{"::cuda::std::equal_to<{}>", binary_predicate_matcher, "=="};
      case cccl_op_kind_t::CCCL_NOT_EQUAL_TO:
        return well_known_description{"::cuda::std::not_equal_to<{}>", binary_predicate_matcher, "!="};
      case cccl_op_kind_t::CCCL_GREATER:
        return well_known_description{"::cuda::std::greater<{}>", binary_predicate_matcher, ">"};
      case cccl_op_kind_t::CCCL_LESS:
        return well_known_description{"::cuda::std::less<{}>", binary_predicate_matcher, "<"};
      case cccl_op_kind_t::CCCL_GREATER_EQUAL:
        return well_known_description{"::cuda::std::greater_equal<{}>", binary_predicate_matcher, ">="};
      case cccl_op_kind_t::CCCL_LESS_EQUAL:
        return well_known_description{"::cuda::std::less_equal<{}>", binary_predicate_matcher, "<="};
      case cccl_op_kind_t::CCCL_LOGICAL_AND:
        return well_known_description{"::cuda::std::logical_and<{}>", binary_predicate_matcher, "&&"};
      case cccl_op_kind_t::CCCL_LOGICAL_OR:
        return well_known_description{"::cuda::std::logical_or<{}>", binary_predicate_matcher, "||"};
      case cccl_op_kind_t::CCCL_LOGICAL_NOT:
        return well_known_description{"::cuda::std::logical_not<{}>", unary_predicate_matcher, "!"};
      case cccl_op_kind_t::CCCL_BIT_AND:
        return well_known_description{"::cuda::std::bit_and<{}>", binary_matcher, "&"};
      case cccl_op_kind_t::CCCL_BIT_OR:
        return well_known_description{"::cuda::std::bit_or<{}>", binary_matcher, "|"};
      case cccl_op_kind_t::CCCL_BIT_XOR:
        return well_known_description{"::cuda::std::bit_xor<{}>", binary_matcher, "^"};
      case cccl_op_kind_t::CCCL_BIT_NOT:
        return well_known_description{"::cuda::std::bit_not<{}>", unary_matcher, "~"};
      case cccl_op_kind_t::CCCL_IDENTITY:
        return well_known_description{"::cuda::std::identity", unary_matcher};
      case cccl_op_kind_t::CCCL_NEGATE:
        return well_known_description{"::cuda::std::negate<{}>", unary_matcher, "-"};
      case cccl_op_kind_t::CCCL_MINIMUM:
        return well_known_description{"::cuda::minimum<{}>", binary_matcher};
      case cccl_op_kind_t::CCCL_MAXIMUM:
        return well_known_description{"::cuda::maximum<{}>", binary_matcher};
      default:
        throw std::runtime_error("c.parallel: invalid well-known operation queried.");
    }
  }

  template <typename, typename... Args>
  static cuda::std::optional<specialization> special(cccl_op_t operation, cccl_type_info ret, Args... arguments)
  {
    auto entry = well_known_operation_description(operation.type);
    if (!entry)
    {
      return cuda::std::nullopt;
    }

    cccl_type_enum type_info_table[] = {ret.type, arguments.type...};
    auto builder                     = entry->check(cuda::std::span(type_info_table), std::format(entry->name, +""));

    std::string aux = "#include <cuda/std/functional>\n";
    if (entry->symbol)
    {
      if (((arguments.type == cccl_type_enum::CCCL_STORAGE) || ...))
      {
        std::string type_names[] = {cccl_type_enum_to_name(ret.type), cccl_type_enum_to_name(arguments.type)...};
        auto type_name_views     = [&]<auto... Is>(std::index_sequence<Is...>) {
          return std::array<std::string_view, sizeof...(Is)>{{type_names[Is]...}};
        }(std::make_index_sequence<1 + sizeof...(arguments)>());
        aux += builder(cuda::std::span(type_name_views), *entry->symbol, operation.name);
      }
      else if (ret.type == cccl_type_enum::CCCL_STORAGE)
      {
        return cuda::std::nullopt;
      }
    }

    return specialization{std::format(entry->name, cccl_type_enum_to_name(type_info_table[1]).c_str()), aux};
  }
#endif
};

struct binary_user_operation_traits
{
  static const constexpr auto name = "binary_user_operation_traits::type";
  template <typename Tag, cccl_op_t_mapping Operation, cccl_type_info_mapping ValueT>
  using type = user_operation_traits::type<Tag, Operation, ValueT, ValueT, ValueT>;

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
  template <typename Tag, typename... Args>
  static cuda::std::optional<specialization> special(cccl_op_t operation, cccl_type_info arg_t)
  {
    return user_operation_traits::special<Tag>(operation, arg_t, arg_t, arg_t);
  }
#endif
};

struct binary_user_predicate_traits
{
  static const constexpr auto name = "binary_user_predicate_traits::type";
  template <typename Tag, cccl_op_t_mapping Operation, cccl_type_info_mapping ValueT>
  using type = user_operation_traits::type<Tag, Operation, cccl_type_info_mapping<bool>{}, ValueT, ValueT>;

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
  template <typename Tag, typename... Args>
  static cuda::std::optional<specialization> special(cccl_op_t operation, cccl_type_info arg_t)
  {
    return user_operation_traits::special<Tag>(operation, arg_t, arg_t, arg_t);
  }
#endif
};
