//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>
#include <variant>
#include <vector>

#include <cccl/c/types.h>
#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

namespace hostjit::codegen
{
// Tags for non-cccl arguments (no runtime data, just control code generation)
struct temp_storage_t
{};
struct temp_bytes_t
{};
// num_items_t carries a name so the same tag type can express num_segments,
// num_needles, etc. — each becomes its own unsigned long long parameter.
struct num_items_t
{
  const char* name = "num_items";
};
struct stream_t
{};

inline constexpr temp_storage_t temp_storage{};
inline constexpr temp_bytes_t temp_bytes{};
inline constexpr num_items_t num_items{};
inline constexpr num_items_t num_segments{"num_segments"};
inline constexpr num_items_t num_needles{"num_needles"};
inline constexpr num_items_t num_haystack{"num_haystack"};
inline constexpr stream_t stream{};

// Direction wrappers for iterators (cccl_iterator_t doesn't encode direction)
struct input_t
{
  cccl_iterator_t it;
};
struct output_t
{
  cccl_iterator_t it;
};

inline input_t in(cccl_iterator_t it)
{
  return {it};
}
inline output_t out(cccl_iterator_t it)
{
  return {it};
}

// cmp_t: wraps a cccl_op_t that should generate a comparison functor
// (bool operator()(const T&, const T&)) rather than the default binary reduce
// functor (T operator()(T, T)).  Use cmp(op) where sort/search operators go.
struct cmp_t
{
  cccl_op_t op;
};
inline cmp_t cmp(cccl_op_t op)
{
  return {op};
}

// future_val_t: the init value lives on the device at runtime.  Generates
// cub::FutureValue<accum_t>(static_cast<accum_t*>(param)) in the CUB call.
// Carries type info so find_accum_type can resolve accum_t correctly.
struct future_val_t
{
  cccl_type_info type;
};
inline future_val_t future_val(cccl_type_info t)
{
  return {t};
}

// unary_op_t: wraps a cccl_op_t used as a unary transform operator (T -> U).
// Carries the input/output type info so the functor can be typed correctly.
struct unary_op_t
{
  cccl_op_t op;
  cccl_type_info in_type;
  cccl_type_info out_type;
};
inline unary_op_t unary_op(cccl_op_t op, cccl_type_info in_t, cccl_type_info out_t)
{
  return {op, in_t, out_t};
}

// force_accum_type_t: overrides the accumulator type resolved by find_accum_type.
// Use when the natural accum type (first input) differs from the desired type.
// Generates no code — only influences type resolution.
struct force_accum_type_t
{
  cccl_type_info type;
};
inline force_accum_type_t force_accum_type(cccl_type_info t)
{
  return {t};
}

// pred(): shorthand for a unary bool predicate operator (e.g. for partition).
// Equivalent to unary_op with out_type = bool.
// Generates: bool operator()(const item_t& a) const { ... }
inline unary_op_t pred(cccl_op_t op, cccl_type_info item_t)
{
  return {op, item_t, cccl_type_info{sizeof(bool), alignof(bool), CCCL_BOOLEAN}};
}

// typed_scalar_t: a by-value scalar of any cccl-known type, passed into the
// JIT wrapper as a host pointer and memcpy'd onto the stack before the CUB
// call. Use when the CUB API takes a small POD by value (e.g. radix_sort's
// `int begin_bit`, histogram's `int num_levels` / `level_t lower_level`).
// The caller supplies a void* host pointer to the value at the corresponding
// run-time arg position.
struct typed_scalar_t
{
  cccl_type_info type;
  const char* name;
};
inline typed_scalar_t typed_scalar(cccl_type_info t, const char* name)
{
  return {t, name};
}

// env_stream_t: variant of stream_t that emits a cuda::std::execution::env
// wrapping a cuda::stream_ref instead of a bare cudaStream_t. Use with CUB
// algorithms that take an env (so CUB manages temp storage internally via
// the env's memory_resource — caller doesn't have to thread it through).
struct env_stream_t
{};
inline constexpr env_stream_t env_stream{};

// for_each_op_t: wraps a cccl_op_t with c.parallel's void op(T*) contract
// into the void op(T&) functor that cub::DeviceFor::ForEachN expects.
struct for_each_op_t
{
  cccl_op_t op;
};
inline for_each_op_t for_each_op(cccl_op_t op)
{
  return {op};
}

// double_buffer_t: constructs a cub::DoubleBuffer<elem_t> from two host
// pointers (one "in" buffer, one "out" buffer) and passes it to the CUB call.
// Used by the DoubleBuffer overloads of DeviceRadixSort / DeviceSegmentedSort
// where the caller is willing to let CUB swap buffers and report the final
// location via the buffer's `selector` member. var_name controls the C++ name
// of the generated local — pair a `selector_out_t` with the same name to read
// `<var_name>.selector` after the call.
struct double_buffer_t
{
  cccl_iterator_t in_it;
  cccl_iterator_t out_it;
  const char* var_name;
};
inline double_buffer_t double_buffer(cccl_iterator_t in_it, cccl_iterator_t out_it, const char* var_name = "d_buffer")
{
  return {in_it, out_it, var_name};
}

// selector_out_t: emits a `void* selector_out` parameter and, after the CUB
// call, writes `<buffer_var_name>.selector` to it. Must be paired with a
// double_buffer_t whose var_name matches.
struct selector_out_t
{
  const char* buffer_var_name;
};
inline selector_out_t selector_out(const char* buffer_var_name = "d_buffer")
{
  return {buffer_var_name};
}

// Argument variant: everything that can appear in .with()
using Arg = std::variant<
  temp_storage_t,
  temp_bytes_t,
  num_items_t,
  stream_t,
  env_stream_t,
  input_t,
  output_t,
  cccl_op_t,
  cmp_t,
  unary_op_t,
  for_each_op_t,
  double_buffer_t,
  selector_out_t,
  future_val_t,
  cccl_value_t,
  force_accum_type_t,
  typed_scalar_t>;

// Result of a successful single-function compilation.
struct CubCallResult
{
  JITCompiler* compiler; // caller takes ownership
  void* fn_ptr; // the exported function
  std::vector<char> cubin; // for SASS inspection
};

// Result of a successful multi-function compilation (one TU, N functions).
struct MultiCubCallResult
{
  JITCompiler* compiler; // caller takes ownership; one compiler for the whole TU
  std::vector<char> cubin; // single cubin for the whole TU
  std::vector<void*> fn_ptrs; // exported functions in the same order as the input CubCalls
};

class CubCall
{
public:
  // Start building: specify the CUB header to include.
  static CubCall from(const char* include_header);

  // Specify the CUB function to call (e.g., "cub::DeviceReduce::Reduce").
  CubCall& run(const char* cub_function);

  // Optionally override the exported function name (default: "cccl_jit_fn").
  CubCall& name(const char* export_name);

  // Add arguments in CUB call order. Each argument is dispatched by type.
  template <typename... Args>
  CubCall& with(Args&&... args)
  {
    (args_.emplace_back(Arg{std::forward<Args>(args)}), ...);
    return *this;
  }

  // Wrap all input iterators in cuda::std::make_tuple() in the generated CUB call.
  // Required for cub::DeviceTransform::Transform with multiple inputs.
  CubCall& use_tuple_inputs()
  {
    tuple_inputs_ = true;
    return *this;
  }

  // Generate the complete CUDA source string (useful for debugging).
  std::string source() const;

  // Compile the generated source and return the function pointer.
  CubCallResult compile(
    int cc_major,
    int cc_minor,
    cccl_build_config* config     = nullptr,
    const char* ctk_path          = nullptr,
    const char* cccl_include_path = nullptr) const;

  // Compile multiple CubCalls into a single translation unit. One Clang
  // invocation, one cubin, one JITCompiler; each function is dlsym'd by its
  // .name(...) and returned in the input order. All CubCalls must share the
  // same CUB include header (.from(...)). Per-function preambles are isolated
  // inside `namespace fn_<i> { ... }` blocks; extern "C" symbols escape the
  // namespace and stay globally dlsym-able.
  static MultiCubCallResult compile(
    std::initializer_list<CubCall> calls,
    int cc_major,
    int cc_minor,
    cccl_build_config* config     = nullptr,
    const char* ctk_path          = nullptr,
    const char* cccl_include_path = nullptr);

private:
  std::string include_;
  std::string cub_function_;
  std::string fn_name_ = "cccl_jit_fn";
  std::vector<Arg> args_;
  bool tuple_inputs_ = false;

  // Internal: just the per-function body (preamble + function defn), no
  // shared #includes and no EXPORT macro. Used by the multi-compile path to
  // wrap N bodies in N namespaces under a single shared include block.
  std::string body() const;

  // Internal: walk args_ and register any user-op / iterator bitcode with
  // the given collector. Factored out so the multi-compile path can share
  // one collector across several CubCalls.
  void collect_bitcode(class BitcodeCollector& bitcode, int& op_idx, int& in_idx, int& out_idx) const;
};
} // namespace hostjit::codegen
