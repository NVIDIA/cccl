//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/reduce.h>
#include <cccl/c/scan.h>
#include <nvrtc.h>

static std::string inspect_sass(const void* cubin, size_t cubin_size)
{
  namespace fs = std::filesystem;

  fs::path temp_dir = fs::temp_directory_path();

  fs::path temp_in_filename  = temp_dir / "temp_in_file.cubin";
  fs::path temp_out_filename = temp_dir / "temp_out_file.sass";

  std::ofstream temp_in_file(temp_in_filename, std::ios::binary);
  if (!temp_in_file)
  {
    throw std::runtime_error("Failed to create temporary file.");
  }

  temp_in_file.write(static_cast<const char*>(cubin), cubin_size);
  temp_in_file.close();

  std::string command = "nvdisasm -gi ";
  command += temp_in_filename;
  command += " > ";
  command += temp_out_filename;

  int exec_code = std::system(command.c_str());

  if (!fs::remove(temp_in_filename))
  {
    throw std::runtime_error("Failed to remove temporary file.");
  }

  if (exec_code != 0)
  {
    throw std::runtime_error("Failed to execute command.");
  }

  std::ifstream temp_out_file(temp_out_filename, std::ios::binary);
  if (!temp_out_file)
  {
    throw std::runtime_error("Failed to create temporary file.");
  }

  const std::string sass{std::istreambuf_iterator<char>(temp_out_file), std::istreambuf_iterator<char>()};
  if (!fs::remove(temp_out_filename))
  {
    throw std::runtime_error("Failed to remove temporary file.");
  }

  return sass;
}

static std::string compile(const std::string& source)
{
  // compile source to LTO-IR using nvrtc

  nvrtcProgram prog;
  REQUIRE(NVRTC_SUCCESS == nvrtcCreateProgram(&prog, source.c_str(), "op.cu", 0, nullptr, nullptr));

  const char* options[] = {"--std=c++17", "-rdc=true", "-dlto"};

  if (nvrtcCompileProgram(prog, 3, options) != NVRTC_SUCCESS)
  {
    size_t log_size{};
    REQUIRE(NVRTC_SUCCESS == nvrtcGetProgramLogSize(prog, &log_size));
    std::unique_ptr<char[]> log{new char[log_size]};
    REQUIRE(NVRTC_SUCCESS == nvrtcGetProgramLog(prog, log.get()));
    printf("%s\r\n", log.get());
    REQUIRE(false);
  }

  std::size_t ltoir_size{};
  REQUIRE(NVRTC_SUCCESS == nvrtcGetLTOIRSize(prog, &ltoir_size));

  std::unique_ptr<char[]> ltoir(new char[ltoir_size]);

  REQUIRE(NVRTC_SUCCESS == nvrtcGetLTOIR(prog, ltoir.get()));
  REQUIRE(NVRTC_SUCCESS == nvrtcDestroyProgram(&prog));

  return std::string(ltoir.get(), ltoir_size);
}

template <class T>
std::vector<T> generate(std::size_t num_items)
{
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<T> dist{T{1}, T{42}};
  std::vector<T> vec(num_items);
  std::generate(vec.begin(), vec.end(), [&]() {
    return dist(mersenne_engine);
  });
  return vec;
}

template <class T>
std::vector<T> make_shuffled_sequence(std::size_t num_items)
{
  std::vector<T> sequence(num_items);
  std::iota(sequence.begin(), sequence.end(), T(0));
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::shuffle(sequence.begin(), sequence.end(), mersenne_engine);
  return sequence;
}

template <class T>
cccl_type_info get_type_info()
{
  cccl_type_info info;
  info.size      = sizeof(T);
  info.alignment = alignof(T);

  if constexpr (std::is_same_v<T, char> || (std::is_integral_v<T> && std::is_signed_v<T> && sizeof(T) == sizeof(char)))
  {
    info.type = cccl_type_enum::CCCL_INT8;
  }
  else if constexpr (std::is_same_v<T, uint8_t>
                     || (std::is_integral_v<T> && std::is_unsigned_v<T> && sizeof(T) == sizeof(char)
                         && !std::is_same_v<T, bool>) )
  {
    info.type = cccl_type_enum::CCCL_UINT8;
  }
  else if constexpr (std::is_same_v<T, int16_t>
                     || (std::is_integral_v<T> && std::is_signed_v<T> && sizeof(T) == sizeof(int16_t)))
  {
    info.type = cccl_type_enum::CCCL_INT16;
  }
  else if constexpr (std::is_same_v<T, uint16_t>
                     || (std::is_integral_v<T> && std::is_unsigned_v<T> && sizeof(T) == sizeof(int16_t)))
  {
    info.type = cccl_type_enum::CCCL_UINT16;
  }
  else if constexpr (std::is_same_v<T, int32_t>
                     || (std::is_integral_v<T> && std::is_signed_v<T> && sizeof(T) == sizeof(int32_t)))
  {
    info.type = cccl_type_enum::CCCL_INT32;
  }
  else if constexpr (std::is_same_v<T, uint32_t>
                     || (std::is_integral_v<T> && std::is_unsigned_v<T> && sizeof(T) == sizeof(int32_t)))
  {
    info.type = cccl_type_enum::CCCL_UINT32;
  }
  else if constexpr (std::is_same_v<T, int64_t>
                     || (std::is_integral_v<T> && std::is_signed_v<T> && sizeof(T) == sizeof(int64_t)))
  {
    info.type = cccl_type_enum::CCCL_INT64;
  }
  else if constexpr (std::is_same_v<T, uint64_t>
                     || (std::is_integral_v<T> && std::is_unsigned_v<T> && sizeof(T) == sizeof(int64_t)))
  {
    info.type = cccl_type_enum::CCCL_UINT64;
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    info.type = cccl_type_enum::CCCL_FLOAT32;
  }
  else if constexpr (std::is_same_v<T, double>)
  {
    info.type = cccl_type_enum::CCCL_FLOAT64;
  }
  else if constexpr (!std::is_integral_v<T>)
  {
    info.type = cccl_type_enum::CCCL_STORAGE;
  }
  else
  {
    static_assert(false, "Unsupported type");
  }

  return info;
}

static std::string get_reduce_op(cccl_type_enum t)
{
  switch (t)
  {
    case cccl_type_enum::CCCL_INT8:
      return "extern \"C\" __device__ char op(char a, char b) { return a + b; }";
    case cccl_type_enum::CCCL_INT32:
      return "extern \"C\" __device__ int op(int a, int b) { return a + b; }";
    case cccl_type_enum::CCCL_UINT32:
      return "extern \"C\" __device__ unsigned int op(unsigned int a, unsigned int b) { return a + b; }";
    case cccl_type_enum::CCCL_INT64:
      return "extern \"C\" __device__ long long op(long long a, long long b) { return a + b; }";
    case cccl_type_enum::CCCL_UINT64:
      return "extern \"C\" __device__ unsigned long long op(unsigned long long a, unsigned long long b) { "
             " return a + b; "
             "}";
    case cccl_type_enum::CCCL_FLOAT32:
      return "extern \"C\" __device__ float op(float a, float b) { return a + b; }";
    case cccl_type_enum::CCCL_FLOAT64:
      return "extern \"C\" __device__ double op(double a, double b) { return a + b; }";
    default:
      throw std::runtime_error("Unsupported type");
  }
  return "";
}

static std::string get_for_op(cccl_type_enum t)
{
  switch (t)
  {
    case cccl_type_enum::CCCL_INT8:
      return "extern \"C\" __device__ void op(char* a) {(*a)++;}";
    case cccl_type_enum::CCCL_INT32:
      return "extern \"C\" __device__ void op(int* a) {(*a)++;}";
    case cccl_type_enum::CCCL_UINT32:
      return "extern \"C\" __device__ void op(unsigned int* a) {(*a)++;}";
    case cccl_type_enum::CCCL_INT64:
      return "extern \"C\" __device__ void op(long long* a) {(*a)++;}";
    case cccl_type_enum::CCCL_UINT64:
      return "extern \"C\" __device__ void op(unsigned long long* a) {(*a)++;}";
    default:
      throw std::runtime_error("Unsupported type");
  }
  return "";
}

static std::string get_merge_sort_op(cccl_type_enum t)
{
  switch (t)
  {
    case cccl_type_enum::CCCL_INT8:
      return "extern \"C\" __device__ bool op(char lhs, char rhs) { return lhs < rhs; }";
    case cccl_type_enum::CCCL_UINT8:
      return "extern \"C\" __device__ bool op(unsigned char lhs, unsigned char rhs) { return lhs < rhs; }";
    case cccl_type_enum::CCCL_INT16:
      return "extern \"C\" __device__ bool op(short lhs, short rhs) { return lhs < rhs; }";
    case cccl_type_enum::CCCL_UINT16:
      return "extern \"C\" __device__ bool op(unsigned short lhs, unsigned short rhs) { return lhs < rhs; }";
    case cccl_type_enum::CCCL_INT32:
      return "extern \"C\" __device__ bool op(int lhs, int rhs) { return lhs < rhs; }";
    case cccl_type_enum::CCCL_UINT32:
      return "extern \"C\" __device__ bool op(unsigned int lhs, unsigned int rhs) { return lhs < rhs; }";
    case cccl_type_enum::CCCL_INT64:
      return "extern \"C\" __device__ bool op(long long lhs, long long rhs) { return lhs < rhs; }";
    case cccl_type_enum::CCCL_UINT64:
      return "extern \"C\" __device__ bool op(unsigned long long lhs, unsigned long long rhs) { return lhs < rhs; }";
    case cccl_type_enum::CCCL_FLOAT32:
      return "extern \"C\" __device__ bool op(float lhs, float rhs) { return lhs < rhs; }";
    case cccl_type_enum::CCCL_FLOAT64:
      return "extern \"C\" __device__ bool op(double lhs, double rhs) { return lhs < rhs; }";

    default:
      throw std::runtime_error("Unsupported type");
  }
  return "";
}

static std::string get_unique_by_key_op(cccl_type_enum t)
{
  switch (t)
  {
    case cccl_type_enum::CCCL_INT8:
      return "extern \"C\" __device__ bool op(char lhs, char rhs) { return lhs == rhs; }";
    case cccl_type_enum::CCCL_UINT8:
      return "extern \"C\" __device__ bool op(unsigned char lhs, unsigned char rhs) { return lhs == rhs; }";
    case cccl_type_enum::CCCL_INT16:
      return "extern \"C\" __device__ bool op(short lhs, short rhs) { return lhs == rhs; }";
    case cccl_type_enum::CCCL_UINT16:
      return "extern \"C\" __device__ bool op(unsigned short lhs, unsigned short rhs) { return lhs == rhs; }";
    case cccl_type_enum::CCCL_INT32:
      return "extern \"C\" __device__ bool op(int lhs, int rhs) { return lhs == rhs; }";
    case cccl_type_enum::CCCL_UINT32:
      return "extern \"C\" __device__ bool op(unsigned int lhs, unsigned int rhs) { return lhs == rhs; }";
    case cccl_type_enum::CCCL_INT64:
      return "extern \"C\" __device__ bool op(long long lhs, long long rhs) { return lhs == rhs; }";
    case cccl_type_enum::CCCL_UINT64:
      return "extern \"C\" __device__ bool op(unsigned long long lhs, unsigned long long rhs) { return lhs == rhs; }";
    case cccl_type_enum::CCCL_FLOAT32:
      return "extern \"C\" __device__ bool op(float lhs, float rhs) { return lhs == rhs; }";
    case cccl_type_enum::CCCL_FLOAT64:
      return "extern \"C\" __device__ bool op(double lhs, double rhs) { return lhs == rhs; }";
    default:
      throw std::runtime_error("Unsupported type");
  }
  return "";
}

static std::string get_unary_op(cccl_type_enum t)
{
  switch (t)
  {
    case cccl_type_enum::CCCL_INT8:
      return "extern \"C\" __device__ char op(char a) { return 2 * a; }";
    case cccl_type_enum::CCCL_INT32:
      return "extern \"C\" __device__ int op(int a) { return 2 * a; }";
    case cccl_type_enum::CCCL_UINT32:
      return "extern \"C\" __device__ unsigned int op(unsigned int a) { return 2 * a; }";
    case cccl_type_enum::CCCL_INT64:
      return "extern \"C\" __device__ long long op(long long a) { return 2 * a; }";
    case cccl_type_enum::CCCL_UINT64:
      return "extern \"C\" __device__ unsigned long long op(unsigned long long a) { "
             " return 2 * a; "
             "}";
    case cccl_type_enum::CCCL_FLOAT32:
      return "extern \"C\" __device__ float op(float a) { return 2 * a; }";
    case cccl_type_enum::CCCL_FLOAT64:
      return "extern \"C\" __device__ double op(double a) { return 2 * a; }";
    default:
      throw std::runtime_error("Unsupported type");
  }
  return "";
}

static std::string get_radix_sort_decomposer_op(cccl_type_enum t)
{
  switch (t)
  {
    case cccl_type_enum::CCCL_INT8:
      return "extern \"C\" __device__ char* op(char* key) { return key; };";
    case cccl_type_enum::CCCL_UINT8:
      return "extern \"C\" __device__ unsigned char* op(unsigned char* key) { return key; };";
    case cccl_type_enum::CCCL_INT16:
      return "extern \"C\" __device__ short* op(short* key) { return key; };";
    case cccl_type_enum::CCCL_UINT16:
      return "extern \"C\" __device__ unsigned short* op(unsigned short* key) { return key; };";
    case cccl_type_enum::CCCL_INT32:
      return "extern \"C\" __device__ int* op(int* key) { return key; };";
    case cccl_type_enum::CCCL_UINT32:
      return "extern \"C\" __device__ unsigned int* op(unsigned int* key) { return key; };";
    case cccl_type_enum::CCCL_INT64:
      return "extern \"C\" __device__ long long* op(long long* key) { return key; };";
    case cccl_type_enum::CCCL_UINT64:
      return "extern \"C\" __device__ unsigned long long* op(unsigned long long* key) { return key; };";
    case cccl_type_enum::CCCL_FLOAT32:
      return "extern \"C\" __device__ float* op(float* key) { return key; };";
    case cccl_type_enum::CCCL_FLOAT64:
      return "extern \"C\" __device__ double* op(double* key) { return key; };";

    default:
      throw std::runtime_error("Unsupported type");
  }
  return "";
}

std::string type_enum_to_name(cccl_type_enum type)
{
  switch (type)
  {
    case cccl_type_enum::CCCL_INT8:
      return "::cuda::std::int8_t";
    case cccl_type_enum::CCCL_INT16:
      return "::cuda::std::int16_t";
    case cccl_type_enum::CCCL_INT32:
      return "::cuda::std::int32_t";
    case cccl_type_enum::CCCL_INT64:
      return "::cuda::std::int64_t";
    case cccl_type_enum::CCCL_UINT8:
      return "::cuda::std::uint8_t";
    case cccl_type_enum::CCCL_UINT16:
      return "::cuda::std::uint16_t";
    case cccl_type_enum::CCCL_UINT32:
      return "::cuda::std::uint32_t";
    case cccl_type_enum::CCCL_UINT64:
      return "::cuda::std::uint64_t";
    case cccl_type_enum::CCCL_FLOAT32:
      return "float";
    case cccl_type_enum::CCCL_FLOAT64:
      return "double";

    default:
      throw std::runtime_error("Unsupported type");
  }

  return "";
}

template <class T>
struct pointer_t
{
  T* ptr{};
  size_t size{};

  pointer_t(int num_items)
  {
    REQUIRE(cudaSuccess == cudaMalloc(&ptr, num_items * sizeof(T)));
    size = num_items;
  }

  pointer_t(const std::vector<T>& vec)
  {
    REQUIRE(cudaSuccess == cudaMalloc(&ptr, vec.size() * sizeof(T)));
    REQUIRE(cudaSuccess == cudaMemcpy(ptr, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice));
    size = vec.size();
  }

  pointer_t()
      : ptr(nullptr)
      , size(0)
  {}

  ~pointer_t()
  {
    if (ptr)
    {
      REQUIRE(cudaSuccess == cudaFree(ptr));
      ptr = nullptr;
    }
  }

  T operator[](int i) const
  {
    T value{};
    REQUIRE(cudaSuccess == cudaMemcpy(&value, ptr + i, sizeof(T), cudaMemcpyDeviceToHost));
    return value;
  }

  operator cccl_iterator_t()
  {
    cccl_iterator_t it;
    it.size        = sizeof(T);
    it.alignment   = alignof(T);
    it.type        = cccl_iterator_kind_t::CCCL_POINTER;
    it.state       = ptr;
    it.value_type  = get_type_info<T>();
    it.advance     = {};
    it.dereference = {};
    return it;
  }

  operator std::vector<T>() const
  {
    std::vector<T> vec(size);
    REQUIRE(cudaSuccess == cudaMemcpy(vec.data(), ptr, sizeof(T) * size, cudaMemcpyDeviceToHost));
    return vec;
  }
};

struct operation_t
{
  std::string name;
  std::string code;

  operator cccl_op_t()
  {
    cccl_op_t op;
    op.type       = cccl_op_kind_t::CCCL_STATELESS;
    op.name       = name.c_str();
    op.ltoir      = code.c_str();
    op.ltoir_size = code.size();
    op.size       = 1;
    op.alignment  = 1;
    op.state      = nullptr;
    return op;
  }
};

template <class OpT>
struct stateful_operation_t
{
  OpT op_state;
  std::string name;
  std::string code;

  operator cccl_op_t()
  {
    cccl_op_t op;
    op.type       = cccl_op_kind_t::CCCL_STATEFUL;
    op.size       = sizeof(OpT);
    op.alignment  = alignof(OpT);
    op.state      = &op_state;
    op.name       = name.c_str();
    op.ltoir      = code.c_str();
    op.ltoir_size = code.size();
    return op;
  }
};

static operation_t make_operation(std::string name, std::string code)
{
  return operation_t{name, compile(code)};
}

template <class OpT>
static stateful_operation_t<OpT> make_operation(std::string name, std::string code, OpT op)
{
  return {op, name, compile(code)};
}

template <class ValueT, class StateT>
struct iterator_t
{
  StateT state;
  operation_t advance;
  operation_t dereference;

  operator cccl_iterator_t()
  {
    cccl_iterator_t it;
    it.size        = sizeof(StateT);
    it.alignment   = alignof(StateT);
    it.type        = cccl_iterator_kind_t::CCCL_ITERATOR;
    it.advance     = advance;
    it.dereference = dereference;
    it.value_type  = get_type_info<ValueT>();
    it.state       = &state;
    return it;
  }
};

enum class iterator_kind
{
  INPUT  = 0,
  OUTPUT = 1,
};

template <typename T>
struct random_access_iterator_state_t
{
  T* data;
};

template <typename T>
struct counting_iterator_state_t
{
  T value;
};

template <typename T>
struct constant_iterator_state_t
{
  T value;
};

template <class ValueT, class StateT>
iterator_t<ValueT, StateT> make_iterator(std::string state, operation_t advance, operation_t dereference)
{
  iterator_t<ValueT, StateT> it;
  it.advance     = make_operation(advance.name, state + advance.code);
  it.dereference = make_operation(dereference.name, state + dereference.code);
  return it;
}

template <class ValueT>
iterator_t<ValueT, random_access_iterator_state_t<ValueT>> make_random_access_iterator(
  iterator_kind kind, std::string value_type, std::string prefix = "", std::string transform = "")
{
  std::string iterator_state = std::format("struct state_t {{ {0}* data; }};\n", value_type);

  operation_t advance = {
    std::format("{0}_advance", prefix),
    std::format("extern \"C\" __device__ void {0}_advance(state_t* state, unsigned long long offset) {{\n"
                "  state->data += offset;\n"
                "}}",
                prefix)};

  std::string dereference_method;
  if (kind == iterator_kind::INPUT)
  {
    dereference_method = std::format(
      "extern \"C\" __device__ {1} {0}_dereference(state_t* state) {{\n"
      "  return (*state->data){2};\n"
      "}}",
      prefix,
      value_type,
      transform);
  }
  else
  {
    dereference_method = std::format(
      "extern \"C\" __device__ void {0}_dereference(state_t* state, {1} x) {{\n"
      "  *state->data = x{2};\n"
      "}}",
      prefix,
      value_type,
      transform);
  }

  operation_t dereference = {std::format("{0}_dereference", prefix), dereference_method};

  return make_iterator<ValueT, random_access_iterator_state_t<ValueT>>(iterator_state, advance, dereference);
}

template <class ValueT>
iterator_t<ValueT, counting_iterator_state_t<ValueT>>
make_counting_iterator(std::string value_type, std::string prefix = "")
{
  std::string iterator_state = std::format("struct state_t {{ {0} value; }};\n", value_type);

  operation_t advance = {
    std::format("{0}_advance", prefix),
    std::format("extern \"C\" __device__ void {0}_advance(state_t* state, unsigned long long offset) {{\n"
                "  state->value += offset;\n"
                "}}",
                prefix)};

  operation_t dereference = {
    std::format("{0}_dereference", prefix),
    std::format("extern \"C\" __device__ {1} {0}_dereference(state_t* state) {{ \n"
                "  return state->value;\n"
                "}}",
                prefix,
                value_type)};

  return make_iterator<ValueT, counting_iterator_state_t<ValueT>>(iterator_state, advance, dereference);
}

template <class ValueT>
iterator_t<ValueT, constant_iterator_state_t<ValueT>>
make_constant_iterator(std::string value_type, std::string prefix = "")
{
  std::string iterator_state = std::format("struct state_t {{ {0} value; }};\n", value_type);

  operation_t advance = {
    std::format("{0}_advance", prefix),
    std::format("extern \"C\" __device__ void {0}_advance(state_t* state, unsigned long long offset) {{ }}", prefix)};

  operation_t dereference = {
    std::format("{0}_dereference", prefix),
    std::format("extern \"C\" __device__ {1} {0}_dereference(state_t* state) {{ \n"
                "  return state->value;\n"
                "}}",
                prefix,
                value_type)};

  return make_iterator<ValueT, constant_iterator_state_t<ValueT>>(iterator_state, advance, dereference);
}

template <class ValueT>
iterator_t<ValueT, random_access_iterator_state_t<ValueT>>
make_reverse_iterator(iterator_kind kind, std::string value_type, std::string prefix = "", std::string transform = "")
{
  std::string iterator_state = std::format("struct state_t {{ {0}* data; }};\n", value_type);

  operation_t advance = {
    std::format("{0}_advance", prefix),
    std::format("extern \"C\" __device__ void {0}_advance(state_t* state, unsigned long long offset) {{\n"
                "  state->data -= offset;\n"
                "}}",
                prefix)};

  std::string dereference_method;
  if (kind == iterator_kind::INPUT)
  {
    dereference_method = std::format(
      "extern \"C\" __device__ {1} {0}_dereference(state_t* state) {{\n"
      "  return (*state->data){2};\n"
      "}}",
      prefix,
      value_type,
      transform);
  }
  else
  {
    dereference_method = std::format(
      "extern \"C\" __device__ void {0}_dereference(state_t* state, {1} x) {{\n"
      "  *state->data = x{2};\n"
      "}}",
      prefix,
      value_type,
      transform);
  }

  operation_t dereference = {std::format("{0}_dereference", prefix), dereference_method};

  return make_iterator<ValueT, random_access_iterator_state_t<ValueT>>(iterator_state, advance, dereference);
}

template <class T>
struct value_t
{
  T value;

  value_t(T value)
      : value(value)
  {}

  operator cccl_value_t()
  {
    cccl_value_t v;
    v.type  = get_type_info<T>();
    v.state = &value;
    return v;
  }
};
