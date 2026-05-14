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

// Argument variant: everything that can appear in .with()
using Arg =
  std::variant<temp_storage_t,
               temp_bytes_t,
               num_items_t,
               stream_t,
               input_t,
               output_t,
               cccl_op_t,
               cmp_t,
               unary_op_t,
               future_val_t,
               cccl_value_t,
               force_accum_type_t>;

// Result of a successful compilation.
struct CubCallResult
{
  JITCompiler* compiler; // caller takes ownership
  void* fn_ptr; // the exported function
  std::vector<char> cubin; // for SASS inspection
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

private:
  std::string include_;
  std::string cub_function_;
  std::string fn_name_ = "cccl_jit_fn";
  std::vector<Arg> args_;
  bool tuple_inputs_ = false;
};
} // namespace hostjit::codegen
