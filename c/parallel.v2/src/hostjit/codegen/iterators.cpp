#include <algorithm>
#include <cstddef>
#include <format>

#include <hostjit/codegen/iterators.hpp>
#include <hostjit/codegen/types.hpp>

namespace hostjit::codegen
{
namespace
{
// The iterator struct holds a `long long _delta` lazy-offset field, so its
// natural alignment is at least alignof(long long)==8. C++ rejects alignas
// values smaller than the natural alignment; clamp here so user iterators with
// small `it.alignment` (e.g. 1 for a `char` state) still produce a valid struct.
inline std::size_t struct_alignas(std::size_t it_alignment)
{
  const std::size_t base = it_alignment > 0 ? it_alignment : 1;
  return base < alignof(long long) ? alignof(long long) : base;
}
} // namespace

IteratorCode make_input_iterator(
  cccl_iterator_t it,
  const std::string& value_type_name,
  const std::string& accum_type_name,
  const std::string& struct_name,
  const std::string& var_name,
  const std::string& state_param)
{
  IteratorCode result;
  result.local_var = var_name;

  if (it.type == CCCL_POINTER)
  {
    // For pointer iterators, the element type is value_type.
    // When value_type_name is empty (unknown/struct type), resolve it from the iterator's
    // value_type info to get a correctly-sized storage struct — falling back to accum_t
    // would use the wrong element size if the value type differs from the accumulator.
    std::string elem_type;
    if (value_type_name.empty())
    {
      auto elem_alias = struct_name + "_elem_t";
      elem_type       = resolve_type(it.value_type, elem_alias.c_str(), result.preamble);
    }
    else
    {
      elem_type = value_type_name;
    }
    result.type_name = elem_type + "*";
    result.preamble += std::format("using {} = {}*;\n\n", struct_name, elem_type);
    result.setup_code = std::format("{} {} = static_cast<{}>({}); ", struct_name, var_name, struct_name, state_param);
  }
  else
  {
    // Custom iterator with state + advance + dereference
    const std::string adv_name = (it.advance.name && it.advance.name[0]) ? it.advance.name : (var_name + "_advance");
    const std::string deref_name =
      (it.dereference.name && it.dereference.name[0]) ? it.dereference.name : (var_name + "_dereference");

    auto input_val_type = value_type_name.empty() ? accum_type_name : value_type_name;
    auto val_alias      = var_name + "_value_t";

    result.type_name = struct_name;
    result.preamble  = std::format("using {} = {};\n", val_alias, input_val_type);

    result.preamble += std::format(
      "extern \"C\" __device__ void {}(void* state, const void* offset);\n"
      "extern \"C\" __device__ void {}(const void* state, {}* result);\n\n",
      adv_name,
      deref_name,
      val_alias);

    // Positional args: {0}=struct_name, {1}=val_alias, {2}=it.size, {3}=adv_name, {4}=deref_name, {5}=it.alignment
    //
    // Arithmetic ops (+, +=, ++) are __host__ __device__ so CUB's host
    // dispatch (which does `iter += n` etc.) compiles in the freestanding
    // host pass. They accumulate into `_delta` rather than calling the
    // device-only `advance` bitcode. `operator*` (device-only) applies the
    // accumulated `_delta` to a copy of state via `advance`, then derefs.
    // `alignas({5})` matches the iterator's declared state alignment so the
    // user-supplied advance/dereference (which casts state as a pointer/etc.)
    // sees properly-aligned memory.
    result.preamble += std::format(
      "struct alignas({5}) {0} {{\n"
      "  using value_type = {1};\n"
      "  using difference_type = long long;\n"
      "  using pointer = {1}*;\n"
      "  using reference = {1};\n"
      "  using iterator_category = cuda::std::random_access_iterator_tag;\n"
      "\n"
      "  alignas({5}) char state[{2}];\n"
      "  long long _delta = 0;\n"
      "\n"
      "  __host__ __device__ {0} operator+(difference_type n) const {{\n"
      "    {0} copy = *this;\n"
      "    copy._delta += n;\n"
      "    return copy;\n"
      "  }}\n"
      "  __host__ __device__ {0}& operator+=(difference_type n) {{\n"
      "    _delta += n;\n"
      "    return *this;\n"
      "  }}\n"
      "  __host__ __device__ {0}& operator++() {{ return *this += 1; }}\n"
      "  __host__ __device__ {0}  operator++(int) {{ {0} tmp = *this; ++(*this); return tmp; }}\n"
      "  __host__ __device__ difference_type operator-(const {0}&) const {{ return 0; }}\n"
      "  __device__ {1} operator*() const {{\n"
      "    {0} copy = *this;\n"
      "    if (copy._delta != 0) {{\n"
      "      unsigned long long offset = static_cast<unsigned long long>(copy._delta);\n"
      "      {3}(copy.state, &offset);\n"
      "    }}\n"
      "    {1} result;\n"
      "    {4}(copy.state, &result);\n"
      "    return result;\n"
      "  }}\n"
      "  __device__ {1} operator[](difference_type n) const {{ return *(*this + n); }}\n"
      "  __host__ __device__ bool operator==(const {0}&) const {{ return false; }}\n"
      "  __host__ __device__ bool operator!=(const {0}&) const {{ return true; }}\n"
      "}};\n\n",
      struct_name, // {0}
      val_alias, // {1}
      it.size, // {2}
      adv_name, // {3}
      deref_name, // {4}
      struct_alignas(it.alignment)); // {5}

    result.setup_code = std::format(
      "{} {};\n"
      "    __builtin_memcpy({}.state, {}, {});",
      struct_name,
      var_name,
      var_name,
      state_param,
      it.size);
  }

  return result;
}

IteratorCode make_output_iterator(
  cccl_iterator_t it,
  const std::string& accum_type_name,
  const std::string& struct_name,
  const std::string& var_name,
  const std::string& state_param,
  const std::string& value_type_name)
{
  IteratorCode result;
  result.local_var = var_name;

  // For custom iterators the element type comes from the dereference function so the
  // accum_t fallback is fine; for pointer iterators we resolve the actual value_type
  // below to get the correct element size.
  const std::string elem_type = value_type_name.empty() ? accum_type_name : value_type_name;

  if (it.type == CCCL_POINTER)
  {
    // When value_type_name is empty (unknown/struct type), resolve from the iterator's own
    // value_type info so the element size is correct — not from accum_t which may differ.
    std::string ptr_elem_type;
    if (value_type_name.empty())
    {
      auto elem_alias = struct_name + "_elem_t";
      ptr_elem_type   = resolve_type(it.value_type, elem_alias.c_str(), result.preamble);
    }
    else
    {
      ptr_elem_type = value_type_name;
    }
    result.type_name = ptr_elem_type + "*";
    result.preamble += std::format("using {} = {}*;\n\n", struct_name, ptr_elem_type);
    result.setup_code = std::format("{} {} = static_cast<{}*>({});", struct_name, var_name, ptr_elem_type, state_param);
  }
  else
  {
    const std::string adv_name = (it.advance.name && it.advance.name[0]) ? it.advance.name : (var_name + "_advance");
    const std::string deref_name =
      (it.dereference.name && it.dereference.name[0]) ? it.dereference.name : (var_name + "_dereference");

    auto proxy_name = var_name + "_proxy_t";

    result.type_name = struct_name;
    result.preamble  = std::format(
      "extern \"C\" __device__ void {}(void* state, const void* offset);\n"
       "extern \"C\" __device__ void {}(void* state, const void* value);\n\n",
      adv_name,
      deref_name);

    // The proxy carries a COPY of the iterator state, not a pointer to it.
    // This is critical for indexed writes (output_it[i] = val): operator[] creates
    // a temporary advanced iterator, calls operator* on it, and returns the proxy
    // by value.  After operator[] returns the temporary is destroyed, so a pointer
    // to its state would be dangling.  Storing the state bytes in the proxy itself
    // makes the proxy self-contained and safe across that return.
    // Proxy contains only `char state[N]` so its natural alignment is 1; the
    // struct alignas is the bigger of the iterator's declared alignment and 1.
    const std::size_t proxy_align = it.alignment > 0 ? it.alignment : 1;
    result.preamble += std::format(
      "struct alignas({1}) {0} {{\n"
      "  alignas({1}) char state[{2}];\n"
      "  __device__ void operator=(const {3}& val) {{\n"
      "    {4}(state, &val);\n"
      "  }}\n"
      "}};\n",
      proxy_name, // {0}
      proxy_align, // {1}
      it.size, // {2}
      elem_type, // {3}
      deref_name); // {4}

    // Arithmetic ops (+, +=, ++) are __host__ __device__ so CUB's host
    // dispatch compiles; they accumulate `_delta` instead of calling the
    // device-only `advance` bitcode. operator* (device only) applies the
    // accumulated `_delta` before constructing the proxy.
    result.preamble += std::format(
      "struct alignas({5}) {0} {{\n"
      "  using value_type = {1};\n"
      "  using difference_type = long long;\n"
      "  using pointer = {1}*;\n"
      "  using reference = {2};\n"
      "  using iterator_category = cuda::std::random_access_iterator_tag;\n"
      "\n"
      "  alignas({5}) char state[{3}];\n"
      "  long long _delta = 0;\n"
      "\n"
      "  __host__ __device__ {0} operator+(difference_type n) const {{\n"
      "    {0} copy = *this;\n"
      "    copy._delta += n;\n"
      "    return copy;\n"
      "  }}\n"
      "  __host__ __device__ {0}& operator+=(difference_type n) {{\n"
      "    _delta += n;\n"
      "    return *this;\n"
      "  }}\n"
      "  __host__ __device__ {0}& operator++() {{ return *this += 1; }}\n"
      "  __host__ __device__ {0}  operator++(int) {{ {0} tmp = *this; ++(*this); return tmp; }}\n"
      "  __host__ __device__ difference_type operator-(const {0}&) const {{ return 0; }}\n"
      "  __device__ reference operator*() const {{\n"
      "    {2} proxy;\n"
      "    __builtin_memcpy(proxy.state, state, {3});\n"
      "    if (_delta != 0) {{\n"
      "      unsigned long long offset = static_cast<unsigned long long>(_delta);\n"
      "      {4}(proxy.state, &offset);\n"
      "    }}\n"
      "    return proxy;\n"
      "  }}\n"
      "  __device__ reference operator[](difference_type n) const {{ return *(*this + n); }}\n"
      "}};\n\n",
      struct_name, // {0}
      elem_type, // {1}
      proxy_name, // {2}
      it.size, // {3}
      adv_name, // {4}
      struct_alignas(it.alignment)); // {5}

    result.setup_code = std::format(
      "{} {};\n"
      "    __builtin_memcpy({}.state, {}, {});",
      struct_name,
      var_name,
      var_name,
      state_param,
      it.size);
  }

  return result;
}
} // namespace hostjit::codegen
