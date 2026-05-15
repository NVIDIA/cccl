#include <format>

#include <hostjit/codegen/operators.hpp>

namespace hostjit::codegen
{
std::string get_well_known_op_body(cccl_op_kind_t kind, const std::string& type_name)
{
  switch (kind)
  {
    case CCCL_PLUS:
      return std::format("    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; {0}* out = ({0}*)out_ptr;\n"
                         "    *out = *a + *b;\n",
                         type_name);
    case CCCL_MINIMUM:
      return std::format("    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; {0}* out = ({0}*)out_ptr;\n"
                         "    *out = (*a < *b) ? *a : *b;\n",
                         type_name);
    case CCCL_MAXIMUM:
      return std::format("    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; {0}* out = ({0}*)out_ptr;\n"
                         "    *out = (*a > *b) ? *a : *b;\n",
                         type_name);
    case CCCL_BIT_AND:
      return std::format("    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; {0}* out = ({0}*)out_ptr;\n"
                         "    *out = *a & *b;\n",
                         type_name);
    case CCCL_BIT_OR:
      return std::format("    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; {0}* out = ({0}*)out_ptr;\n"
                         "    *out = *a | *b;\n",
                         type_name);
    case CCCL_BIT_XOR:
      return std::format("    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; {0}* out = ({0}*)out_ptr;\n"
                         "    *out = *a ^ *b;\n",
                         type_name);
    case CCCL_MULTIPLIES:
      return std::format("    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; {0}* out = ({0}*)out_ptr;\n"
                         "    *out = *a * *b;\n",
                         type_name);
    case CCCL_LESS:
      return std::format("    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; bool* out = (bool*)out_ptr;\n"
                         "    *out = *a < *b;\n",
                         type_name);
    case CCCL_GREATER:
      return std::format("    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; bool* out = (bool*)out_ptr;\n"
                         "    *out = *a > *b;\n",
                         type_name);
    default:
      return "";
  }
}

namespace
{
std::string
generate_op_source(cccl_op_t op, const std::string& accum_type, bool has_bitcode, bool is_stateful, bool is_comparison)
{
  const std::string op_name = (op.name && op.name[0]) ? op.name : "user_op";
  std::string src;

  if (op.code_type == CCCL_OP_CPP_SOURCE && op.code && op.code_size > 0)
  {
    // Embed C++ source directly
    src += std::string(op.code, op.code_size) + "\n\n";
  }
  else if (has_bitcode)
  {
    // Extern declaration for bitcode-linked operation
    if (is_stateful)
    {
      src += std::format("extern \"C\" __device__ void {}(void* state, void* a_ptr, void* b_ptr, void* out_ptr);\n\n",
                         op_name);
    }
    else
    {
      src += std::format("extern \"C\" __device__ void {}(void* a_ptr, void* b_ptr, void* out_ptr);\n\n", op_name);
    }
  }
  else if (op.type >= CCCL_PLUS && op.type <= CCCL_MAXIMUM)
  {
    // Well-known operation - generate inline
    src += std::format("extern \"C\" __device__ void {}(void* a_ptr, void* b_ptr, void* out_ptr) {{\n", op_name);
    src += get_well_known_op_body(op.type, accum_type);
    src += "}\n\n";
  }

  return src;
}

std::string generate_binary_functor(cccl_op_t op, const std::string& accum_type, const std::string& functor_name)
{
  const std::string op_name = (op.name && op.name[0]) ? op.name : "user_op";
  const bool is_stateful    = (op.type == CCCL_STATEFUL);

  // Templated operator() lets CUB instantiate the functor with whatever
  // element types its kernel deduces (important for binary transform with
  // two differently-typed input iterators). The user's bitcode hop takes
  // void* anyway, so the concrete arg types only need to be addressable.
  if (is_stateful)
  {
    // Embed the user's state bytes inline. When CUB launches a kernel with
    // this functor by value, the bytes ride along in the launch-arg buffer
    // into device constant memory, so the address handed to the user's op
    // (`state_bytes`) is a valid device-side pointer. Storing a host pointer
    // here would crash on first device-side dereference.
    const size_t state_size  = op.size > 0 ? op.size : 1;
    const size_t state_align = op.alignment > 0 ? op.alignment : 1;
    return std::format(
      "struct {0} {{\n"
      "  alignas({3}) unsigned char state_bytes[{4}];\n"
      "  template <typename _A, typename _B>\n"
      "  __host__ __device__ __forceinline__\n"
      "  {1} operator()(const _A& a, const _B& b) const {{\n"
      "    {1} result;\n"
      "    {2}((void*)state_bytes, (void*)&a, (void*)&b, (void*)&result);\n"
      "    return result;\n"
      "  }}\n"
      "}};\n\n",
      functor_name,
      accum_type,
      op_name,
      state_align,
      state_size);
  }
  else
  {
    return std::format(
      "struct {0} {{\n"
      "  template <typename _A, typename _B>\n"
      "  __host__ __device__ __forceinline__\n"
      "  {1} operator()(const _A& a, const _B& b) const {{\n"
      "    {1} result;\n"
      "    {2}((void*)&a, (void*)&b, (void*)&result);\n"
      "    return result;\n"
      "  }}\n"
      "}};\n\n",
      functor_name,
      accum_type,
      op_name);
  }
}

std::string generate_comparison_functor(cccl_op_t op, const std::string& key_type, const std::string& functor_name)
{
  const std::string op_name = (op.name && op.name[0]) ? op.name : "user_op";
  const bool is_stateful    = (op.type == CCCL_STATEFUL);

  if (is_stateful)
  {
    // See generate_binary_functor: state must travel by value via kernel-arg
    // copy, not by host pointer, or the device-side deref crashes.
    const size_t state_size  = op.size > 0 ? op.size : 1;
    const size_t state_align = op.alignment > 0 ? op.alignment : 1;
    return std::format(
      "struct {0} {{\n"
      "  alignas({3}) unsigned char state_bytes[{4}];\n"
      "  __host__ __device__ __forceinline__\n"
      "  bool operator()(const {1}& a, const {2}& b) const {{\n"
      "    bool result;\n"
      "    {5}((void*)state_bytes, (void*)&a, (void*)&b, (void*)&result);\n"
      "    return result;\n"
      "  }}\n"
      "}};\n\n",
      functor_name,
      key_type,
      key_type,
      state_align,
      state_size,
      op_name);
  }
  else
  {
    return std::format(
      "struct {} {{\n"
      "  __host__ __device__ __forceinline__\n"
      "  bool operator()(const {}& a, const {}& b) const {{\n"
      "    bool result;\n"
      "    {}((void*)&a, (void*)&b, (void*)&result);\n"
      "    return result;\n"
      "  }}\n"
      "}};\n\n",
      functor_name,
      key_type,
      key_type,
      op_name);
  }
}

// Returns the cuda::std (or cuda::) functor type string for a well-known op, or nullptr if not well-known.
const char* get_well_known_functor_type(cccl_op_kind_t kind)
{
  switch (kind)
  {
    case CCCL_PLUS:
      return "::cuda::std::plus<>";
    case CCCL_MINUS:
      return "::cuda::std::minus<>";
    case CCCL_MULTIPLIES:
      return "::cuda::std::multiplies<>";
    case CCCL_DIVIDES:
      return "::cuda::std::divides<>";
    case CCCL_MODULUS:
      return "::cuda::std::modulus<>";
    case CCCL_EQUAL_TO:
      return "::cuda::std::equal_to<>";
    case CCCL_NOT_EQUAL_TO:
      return "::cuda::std::not_equal_to<>";
    case CCCL_GREATER:
      return "::cuda::std::greater<>";
    case CCCL_LESS:
      return "::cuda::std::less<>";
    case CCCL_GREATER_EQUAL:
      return "::cuda::std::greater_equal<>";
    case CCCL_LESS_EQUAL:
      return "::cuda::std::less_equal<>";
    case CCCL_BIT_AND:
      return "::cuda::std::bit_and<>";
    case CCCL_BIT_OR:
      return "::cuda::std::bit_or<>";
    case CCCL_BIT_XOR:
      return "::cuda::std::bit_xor<>";
    case CCCL_MINIMUM:
      return "::cuda::minimum<>";
    case CCCL_MAXIMUM:
      return "::cuda::maximum<>";
    default:
      return nullptr;
  }
}

// Returns the C++ operator symbol for a well-known op, or nullptr if none.
const char* get_well_known_op_symbol(cccl_op_kind_t kind)
{
  switch (kind)
  {
    case CCCL_PLUS:
      return "+";
    case CCCL_MINUS:
      return "-";
    case CCCL_MULTIPLIES:
      return "*";
    case CCCL_DIVIDES:
      return "/";
    case CCCL_MODULUS:
      return "%";
    case CCCL_EQUAL_TO:
      return "==";
    case CCCL_NOT_EQUAL_TO:
      return "!=";
    case CCCL_GREATER:
      return ">";
    case CCCL_LESS:
      return "<";
    case CCCL_GREATER_EQUAL:
      return ">=";
    case CCCL_LESS_EQUAL:
      return "<=";
    case CCCL_BIT_AND:
      return "&";
    case CCCL_BIT_OR:
      return "|";
    case CCCL_BIT_XOR:
      return "^";
    default:
      return nullptr;
  }
}

// Generate preamble for a well-known binary op.
// For custom types with user-provided code, declares the extern "C" function
// and generates an operator overload that calls it.
// For primitive types without user code, no preamble is needed.
std::string
generate_well_known_preamble(cccl_op_t op, const std::string& accum_type, bool has_bitcode, bool is_comparison)
{
  const std::string op_name     = (op.name && op.name[0]) ? op.name : "user_op";
  const std::string return_type = is_comparison ? "bool" : accum_type;
  const char* symbol            = get_well_known_op_symbol(op.type);
  bool has_user_code            = has_bitcode || (op.code_type == CCCL_OP_CPP_SOURCE && op.code && op.code_size > 0);

  if (!has_user_code)
  {
    // Pure well-known op on a primitive type — no preamble needed.
    return "";
  }

  std::string src;

  if (op.code_type == CCCL_OP_CPP_SOURCE && op.code && op.code_size > 0)
  {
    // Embed C++ source directly (may contain type definitions).
    src += std::string(op.code, op.code_size) + "\n\n";
  }

  // Declare the extern "C" function from bitcode.
  if (has_bitcode)
  {
    src += std::format("extern \"C\" __device__ void {}(void* a_ptr, void* b_ptr, void* out_ptr);\n\n", op_name);
  }

  // Generate an operator overload that calls the user-provided function,
  // so cuda::std::plus<> (etc.) can use it on custom types.
  if (symbol)
  {
    src += std::format(
      "__device__ {0} operator{1}(const {2}& lhs, const {2}& rhs) {{\n"
      "    {0} ret;\n"
      "    {3}((void*)&lhs, (void*)&rhs, (void*)&ret);\n"
      "    return ret;\n"
      "}}\n\n",
      return_type,
      symbol,
      accum_type,
      op_name);
  }

  return src;
}
} // anonymous namespace

OperatorCode make_binary_op(
  cccl_op_t op,
  const std::string& accum_type,
  const std::string& functor_name,
  const std::string& var_name,
  const std::string& state_param,
  bool has_bitcode)
{
  // For well-known operations, use cuda::std functors directly.
  // For custom types, generate an operator overload that wraps the user-provided function.
  // If the caller provided bitcode, prefer it: the well-known functor (e.g.
  // cuda::std::plus<void>) may not be invocable on the custom value type.
  const char* well_known_type = get_well_known_functor_type(op.type);
  if (well_known_type && !has_bitcode)
  {
    OperatorCode result;
    result.local_var  = var_name;
    result.preamble   = generate_well_known_preamble(op, accum_type, has_bitcode, /*is_comparison=*/false);
    result.setup_code = std::format("{} {}{{}};", well_known_type, var_name);
    return result;
  }

  const bool is_stateful = (op.type == CCCL_STATEFUL);

  OperatorCode result;
  result.local_var = var_name;
  result.preamble  = generate_op_source(op, accum_type, has_bitcode, is_stateful, false);
  result.preamble += generate_binary_functor(op, accum_type, functor_name);

  if (is_stateful)
  {
    const size_t state_size = op.size > 0 ? op.size : 1;
    result.setup_code       = std::format(
      "{0} {1}; __builtin_memcpy({1}.state_bytes, {2}, {3});", functor_name, var_name, state_param, state_size);
  }
  else
  {
    result.setup_code = std::format("{} {};", functor_name, var_name);
  }

  return result;
}

OperatorCode make_unary_op(
  cccl_op_t op,
  const std::string& in_type,
  const std::string& out_type,
  const std::string& functor_name,
  const std::string& var_name,
  const std::string& state_param,
  bool has_bitcode)
{
  // NEGATE and IDENTITY map directly to cuda::std unary functors. If the
  // caller provided bitcode, prefer it — cuda::std::negate<> may not be
  // invocable on the user's custom value type.
  if (op.type == CCCL_NEGATE && !has_bitcode)
  {
    OperatorCode result;
    result.local_var  = var_name;
    result.setup_code = std::format("::cuda::std::negate<> {}{{}};", var_name);
    return result;
  }
  if (op.type == CCCL_IDENTITY && !has_bitcode)
  {
    OperatorCode result;
    result.local_var  = var_name;
    result.setup_code = std::format("::cuda::std::identity {}{{}};", var_name);
    return result;
  }

  const bool is_stateful    = (op.type == CCCL_STATEFUL);
  const std::string op_name = (op.name && op.name[0]) ? op.name : "user_op";

  OperatorCode result;
  result.local_var = var_name;

  // Preamble: extern decl or embedded C++ source
  if (op.code_type == CCCL_OP_CPP_SOURCE && op.code && op.code_size > 0)
  {
    result.preamble += std::string(op.code, op.code_size) + "\n\n";
  }
  else if (has_bitcode)
  {
    if (is_stateful)
    {
      result.preamble +=
        std::format("extern \"C\" __device__ void {}(void* state, void* a_ptr, void* result_ptr);\n\n", op_name);
    }
    else
    {
      result.preamble += std::format("extern \"C\" __device__ void {}(void* a_ptr, void* result_ptr);\n\n", op_name);
    }
  }

  // Functor struct
  if (is_stateful)
  {
    // See generate_binary_functor: state must travel by value via kernel-arg
    // copy, not by host pointer, or the device-side deref crashes.
    const size_t state_size  = op.size > 0 ? op.size : 1;
    const size_t state_align = op.alignment > 0 ? op.alignment : 1;
    result.preamble += std::format(
      "struct {0} {{\n"
      "  alignas({4}) unsigned char state_bytes[{5}];\n"
      "  __host__ __device__ __forceinline__\n"
      "  {1} operator()(const {2}& a) const {{\n"
      "    {3} result;\n"
      "    {6}((void*)state_bytes, (void*)&a, (void*)&result);\n"
      "    return result;\n"
      "  }}\n"
      "}};\n\n",
      functor_name,
      out_type,
      in_type,
      out_type,
      state_align,
      state_size,
      op_name);
    result.setup_code = std::format(
      "{0} {1}; __builtin_memcpy({1}.state_bytes, {2}, {3});", functor_name, var_name, state_param, state_size);
  }
  else
  {
    result.preamble += std::format(
      "struct {} {{\n"
      "  __host__ __device__ __forceinline__\n"
      "  {} operator()(const {}& a) const {{\n"
      "    {} result;\n"
      "    {}((void*)&a, (void*)&result);\n"
      "    return result;\n"
      "  }}\n"
      "}};\n\n",
      functor_name,
      out_type,
      in_type,
      out_type,
      op_name);
    result.setup_code = std::format("{} {};", functor_name, var_name);
  }

  return result;
}

OperatorCode make_comparison_op(
  cccl_op_t op,
  const std::string& key_type,
  const std::string& functor_name,
  const std::string& var_name,
  const std::string& state_param,
  bool has_bitcode)
{
  const char* well_known_type = get_well_known_functor_type(op.type);
  if (well_known_type && !has_bitcode)
  {
    OperatorCode result;
    result.local_var  = var_name;
    result.preamble   = generate_well_known_preamble(op, key_type, has_bitcode, /*is_comparison=*/true);
    result.setup_code = std::format("{} {}{{}};", well_known_type, var_name);
    return result;
  }

  const bool is_stateful = (op.type == CCCL_STATEFUL);

  OperatorCode result;
  result.local_var = var_name;
  result.preamble  = generate_op_source(op, key_type, has_bitcode, is_stateful, true);
  result.preamble += generate_comparison_functor(op, key_type, functor_name);

  if (is_stateful)
  {
    const size_t state_size = op.size > 0 ? op.size : 1;
    result.setup_code       = std::format(
      "{0} {1}; __builtin_memcpy({1}.state_bytes, {2}, {3});", functor_name, var_name, state_param, state_size);
  }
  else
  {
    result.setup_code = std::format("{} {};", functor_name, var_name);
  }

  return result;
}
} // namespace hostjit::codegen
