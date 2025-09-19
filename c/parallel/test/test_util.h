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
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <nvrtc.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/types.h>

inline std::string inspect_sass(const void* cubin, size_t cubin_size)
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

inline std::string compile(const std::string& source)
{
  // compile source to LTO-IR using nvrtc

  nvrtcProgram prog;
  REQUIRE(NVRTC_SUCCESS == nvrtcCreateProgram(&prog, source.c_str(), "op.cu", 0, nullptr, nullptr));

  // TEST_CTK_PATH needed to include cuda_fp16.h
  const char* options[] = {"--std=c++17", "-rdc=true", "-dlto", TEST_CTK_PATH};

  if (nvrtcCompileProgram(prog, 4, options) != NVRTC_SUCCESS)
  {
    size_t log_size{};
    REQUIRE(NVRTC_SUCCESS == nvrtcGetProgramLogSize(prog, &log_size));
    std::vector<char> log(log_size);
    REQUIRE(NVRTC_SUCCESS == nvrtcGetProgramLog(prog, log.data()));
    printf("%s\r\n", log.data());
    REQUIRE(false);
  }

  std::size_t ltoir_size{};
  REQUIRE(NVRTC_SUCCESS == nvrtcGetLTOIRSize(prog, &ltoir_size));

  std::vector<char> ltoir(ltoir_size);

  REQUIRE(NVRTC_SUCCESS == nvrtcGetLTOIR(prog, ltoir.data()));
  REQUIRE(NVRTC_SUCCESS == nvrtcDestroyProgram(&prog));

  return std::string(ltoir.data(), ltoir_size);
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
#if _CCCL_HAS_NVFP16()
  else if constexpr (std::is_same_v<T, __half>)
  {
    info.type = cccl_type_enum::CCCL_FLOAT16;
  }
#endif
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

// TOOD: using more than than one `op` in the same TU will fail because
// of the lack of name mangling. Ditto for all `get_*_op` functions.
inline std::string get_reduce_op(cccl_type_enum t)
{
  switch (t)
  {
    case cccl_type_enum::CCCL_INT8:
      return "extern \"C\" __device__ void op(void* a_void, void* b_void, void* out_void) { "
             "  char* a = reinterpret_cast<char*>(a_void); "
             "  char* b = reinterpret_cast<char*>(b_void); "
             "  char* out = reinterpret_cast<char*>(out_void); "
             "  *out = *a + *b; "
             "}";
    case cccl_type_enum::CCCL_INT32:
      return "extern \"C\" __device__ void op(void* a_void, void* b_void, void* out_void) { "
             "  int* a = reinterpret_cast<int*>(a_void); "
             "  int* b = reinterpret_cast<int*>(b_void); "
             "  int* out = reinterpret_cast<int*>(out_void); "
             "  *out = *a + *b; "
             "}";
    case cccl_type_enum::CCCL_UINT32:
      return "extern \"C\" __device__ void op(void* a_void, void* b_void, void* out_void) { "
             "  unsigned int* a = reinterpret_cast<unsigned int*>(a_void); "
             "  unsigned int* b = reinterpret_cast<unsigned int*>(b_void); "
             "  unsigned int* out = reinterpret_cast<unsigned int*>(out_void); "
             "  *out = *a + *b; "
             "}";
    case cccl_type_enum::CCCL_INT64:
      return "extern \"C\" __device__ void op(void* a_void, void* b_void, void* out_void) { "
             "  long long* a = reinterpret_cast<long long*>(a_void); "
             "  long long* b = reinterpret_cast<long long*>(b_void); "
             "  long long* out = reinterpret_cast<long long*>(out_void); "
             "  *out = *a + *b; "
             "}";
    case cccl_type_enum::CCCL_UINT64:
      return "extern \"C\" __device__ void op(void* a_void, void* b_void, void* out_void) { "
             "  unsigned long long* a = reinterpret_cast<unsigned long long*>(a_void); "
             "  unsigned long long* b = reinterpret_cast<unsigned long long*>(b_void); "
             "  unsigned long long* out = reinterpret_cast<unsigned long long*>(out_void); "
             "  *out = *a + *b; "
             "}";
    case cccl_type_enum::CCCL_FLOAT32:
      return "extern \"C\" __device__ void op(void* a_void, void* b_void, void* out_void) { "
             "  float* a = reinterpret_cast<float*>(a_void); "
             "  float* b = reinterpret_cast<float*>(b_void); "
             "  float* out = reinterpret_cast<float*>(out_void); "
             "  *out = *a + *b; "
             "}";
    case cccl_type_enum::CCCL_FLOAT64:
      return "extern \"C\" __device__ void op(void* a_void, void* b_void, void* out_void) { "
             "  double* a = reinterpret_cast<double*>(a_void); "
             "  double* b = reinterpret_cast<double*>(b_void); "
             "  double* out = reinterpret_cast<double*>(out_void); "
             "  *out = *a + *b; "
             "}";
    case cccl_type_enum::CCCL_FLOAT16:
      return "#include <cuda_fp16.h>\n"
             "extern \"C\" __device__ void op(void* a_void, void* b_void, void* out_void) { "
             "  __half* a = reinterpret_cast<__half*>(a_void); "
             "  __half* b = reinterpret_cast<__half*>(b_void); "
             "  __half* out = reinterpret_cast<__half*>(out_void); "
             "  *out = *a + *b; "
             "}";
    default:
      throw std::runtime_error("Unsupported type");
  }
  return "";
}

inline std::string get_for_op(cccl_type_enum t)
{
  switch (t)
  {
    case cccl_type_enum::CCCL_INT8:
      return "extern \"C\" __device__ void op(void* a_void) { "
             "  char* a = reinterpret_cast<char*>(a_void); "
             "  (*a)++; "
             "}";
    case cccl_type_enum::CCCL_INT32:
      return "extern \"C\" __device__ void op(void* a_void) { "
             "  int* a = reinterpret_cast<int*>(a_void); "
             "  (*a)++; "
             "}";
    case cccl_type_enum::CCCL_UINT32:
      return "extern \"C\" __device__ void op(void* a_void) { "
             "  unsigned int* a = reinterpret_cast<unsigned int*>(a_void); "
             "  (*a)++; "
             "}";
    case cccl_type_enum::CCCL_INT64:
      return "extern \"C\" __device__ void op(void* a_void) { "
             "  long long* a = reinterpret_cast<long long*>(a_void); "
             "  (*a)++; "
             "}";
    case cccl_type_enum::CCCL_UINT64:
      return "extern \"C\" __device__ void op(void* a_void) { "
             "  unsigned long long* a = reinterpret_cast<unsigned long long*>(a_void); "
             "  (*a)++; "
             "}";
    default:
      throw std::runtime_error("Unsupported type");
  }
  return "";
}

inline std::string get_merge_sort_op(cccl_type_enum t)
{
  switch (t)
  {
    case cccl_type_enum::CCCL_INT8:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  char* lhs = reinterpret_cast<char*>(lhs_void); "
             "  char* rhs = reinterpret_cast<char*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs < *rhs; "
             "}";
    case cccl_type_enum::CCCL_UINT8:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  unsigned char* lhs = reinterpret_cast<unsigned char*>(lhs_void); "
             "  unsigned char* rhs = reinterpret_cast<unsigned char*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs < *rhs; "
             "}";
    case cccl_type_enum::CCCL_INT16:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  short* lhs = reinterpret_cast<short*>(lhs_void); "
             "  short* rhs = reinterpret_cast<short*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs < *rhs; "
             "}";
    case cccl_type_enum::CCCL_UINT16:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  unsigned short* lhs = reinterpret_cast<unsigned short*>(lhs_void); "
             "  unsigned short* rhs = reinterpret_cast<unsigned short*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs < *rhs; "
             "}";
    case cccl_type_enum::CCCL_INT32:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  int* lhs = reinterpret_cast<int*>(lhs_void); "
             "  int* rhs = reinterpret_cast<int*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs < *rhs; "
             "}";
    case cccl_type_enum::CCCL_UINT32:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  unsigned int* lhs = reinterpret_cast<unsigned int*>(lhs_void); "
             "  unsigned int* rhs = reinterpret_cast<unsigned int*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs < *rhs; "
             "}";
    case cccl_type_enum::CCCL_INT64:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  long long* lhs = reinterpret_cast<long long*>(lhs_void); "
             "  long long* rhs = reinterpret_cast<long long*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs < *rhs; "
             "}";
    case cccl_type_enum::CCCL_UINT64:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  unsigned long long* lhs = reinterpret_cast<unsigned long long*>(lhs_void); "
             "  unsigned long long* rhs = reinterpret_cast<unsigned long long*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs < *rhs; "
             "}";
    case cccl_type_enum::CCCL_FLOAT32:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  float* lhs = reinterpret_cast<float*>(lhs_void); "
             "  float* rhs = reinterpret_cast<float*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs < *rhs; "
             "}";
    case cccl_type_enum::CCCL_FLOAT64:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  double* lhs = reinterpret_cast<double*>(lhs_void); "
             "  double* rhs = reinterpret_cast<double*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs < *rhs; "
             "}";
    case cccl_type_enum::CCCL_FLOAT16:
      return "#include <cuda_fp16.h>\n"
             "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  __half* lhs = reinterpret_cast<__half*>(lhs_void); "
             "  __half* rhs = reinterpret_cast<__half*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs < *rhs; "
             "}";
    default:
      throw std::runtime_error("Unsupported type");
  }
  return "";
}

inline std::string get_unique_by_key_op(cccl_type_enum t)
{
  switch (t)
  {
    case cccl_type_enum::CCCL_INT8:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  char* lhs = reinterpret_cast<char*>(lhs_void); "
             "  char* rhs = reinterpret_cast<char*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs == *rhs; "
             "}";
    case cccl_type_enum::CCCL_UINT8:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  unsigned char* lhs = reinterpret_cast<unsigned char*>(lhs_void); "
             "  unsigned char* rhs = reinterpret_cast<unsigned char*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs == *rhs; "
             "}";
    case cccl_type_enum::CCCL_INT16:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  short* lhs = reinterpret_cast<short*>(lhs_void); "
             "  short* rhs = reinterpret_cast<short*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs == *rhs; "
             "}";
    case cccl_type_enum::CCCL_UINT16:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  unsigned short* lhs = reinterpret_cast<unsigned short*>(lhs_void); "
             "  unsigned short* rhs = reinterpret_cast<unsigned short*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs == *rhs; "
             "}";
    case cccl_type_enum::CCCL_INT32:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  int* lhs = reinterpret_cast<int*>(lhs_void); "
             "  int* rhs = reinterpret_cast<int*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs == *rhs; "
             "}";
    case cccl_type_enum::CCCL_UINT32:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  unsigned int* lhs = reinterpret_cast<unsigned int*>(lhs_void); "
             "  unsigned int* rhs = reinterpret_cast<unsigned int*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs == *rhs; "
             "}";
    case cccl_type_enum::CCCL_INT64:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  long long* lhs = reinterpret_cast<long long*>(lhs_void); "
             "  long long* rhs = reinterpret_cast<long long*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs == *rhs; "
             "}";
    case cccl_type_enum::CCCL_UINT64:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  unsigned long long* lhs = reinterpret_cast<unsigned long long*>(lhs_void); "
             "  unsigned long long* rhs = reinterpret_cast<unsigned long long*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs == *rhs; "
             "}";
    case cccl_type_enum::CCCL_FLOAT32:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  float* lhs = reinterpret_cast<float*>(lhs_void); "
             "  float* rhs = reinterpret_cast<float*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs == *rhs; "
             "}";
    case cccl_type_enum::CCCL_FLOAT64:
      return "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  double* lhs = reinterpret_cast<double*>(lhs_void); "
             "  double* rhs = reinterpret_cast<double*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs == *rhs; "
             "}";
    case cccl_type_enum::CCCL_FLOAT16:
      return "#include <cuda_fp16.h>\n"
             "extern \"C\" __device__ void op(void* lhs_void, void* rhs_void, void* result_void) { "
             "  __half* lhs = reinterpret_cast<__half*>(lhs_void); "
             "  __half* rhs = reinterpret_cast<__half*>(rhs_void); "
             "  bool* result = reinterpret_cast<bool*>(result_void); "
             "  *result = *lhs == *rhs; "
             "}";
    default:
      throw std::runtime_error("Unsupported type");
  }
  return "";
}

inline std::string get_unary_op(cccl_type_enum t)
{
  switch (t)
  {
    case cccl_type_enum::CCCL_INT8:
      return "extern \"C\" __device__ void op(void* a_void, void* result_void) { "
             "  char* a = reinterpret_cast<char*>(a_void); "
             "  char* result = reinterpret_cast<char*>(result_void); "
             "  *result = 2 * *a; "
             "}";
    case cccl_type_enum::CCCL_INT32:
      return "extern \"C\" __device__ void op(void* a_void, void* result_void) { "
             "  int* a = reinterpret_cast<int*>(a_void); "
             "  int* result = reinterpret_cast<int*>(result_void); "
             "  *result = 2 * *a; "
             "}";
    case cccl_type_enum::CCCL_UINT32:
      return "extern \"C\" __device__ void op(void* a_void, void* result_void) { "
             "  unsigned int* a = reinterpret_cast<unsigned int*>(a_void); "
             "  unsigned int* result = reinterpret_cast<unsigned int*>(result_void); "
             "  *result = 2 * *a; "
             "}";
    case cccl_type_enum::CCCL_INT64:
      return "extern \"C\" __device__ void op(void* a_void, void* result_void) { "
             "  long long* a = reinterpret_cast<long long*>(a_void); "
             "  long long* result = reinterpret_cast<long long*>(result_void); "
             "  *result = 2 * *a; "
             "}";
    case cccl_type_enum::CCCL_UINT64:
      return "extern \"C\" __device__ void op(void* a_void, void* result_void) { "
             "  unsigned long long* a = reinterpret_cast<unsigned long long*>(a_void); "
             "  unsigned long long* result = reinterpret_cast<unsigned long long*>(result_void); "
             "  *result = 2 * *a; "
             "}";
    case cccl_type_enum::CCCL_FLOAT32:
      return "extern \"C\" __device__ void op(void* a_void, void* result_void) { "
             "  float* a = reinterpret_cast<float*>(a_void); "
             "  float* result = reinterpret_cast<float*>(result_void); "
             "  *result = 2 * *a; "
             "}";
    case cccl_type_enum::CCCL_FLOAT64:
      return "extern \"C\" __device__ void op(void* a_void, void* result_void) { "
             "  double* a = reinterpret_cast<double*>(a_void); "
             "  double* result = reinterpret_cast<double*>(result_void); "
             "  *result = 2 * *a; "
             "}";
    case cccl_type_enum::CCCL_FLOAT16:
      return "#include <cuda_fp16.h>\n"
             "extern \"C\" __device__ void op(void* a_void, void* result_void) { "
             "  __half* a = reinterpret_cast<__half*>(a_void); "
             "  __half* result = reinterpret_cast<__half*>(result_void); "
             "  *result = __float2half(2.0f) * (*a); "
             "}";
    default:
      throw std::runtime_error("Unsupported type");
  }
  return "";
}

inline std::string get_radix_sort_decomposer_op(cccl_type_enum t)
{
  switch (t)
  {
    case cccl_type_enum::CCCL_INT8:
      return "extern \"C\" __device__ void* op(void* key_void) { "
             "  char* key = reinterpret_cast<char*>(key_void); "
             "  return key; "
             "};";
    case cccl_type_enum::CCCL_UINT8:
      return "extern \"C\" __device__ void* op(void* key_void) { "
             "  unsigned char* key = reinterpret_cast<unsigned char*>(key_void); "
             "  return key; "
             "};";
    case cccl_type_enum::CCCL_INT16:
      return "extern \"C\" __device__ void* op(void* key_void) { "
             "  short* key = reinterpret_cast<short*>(key_void); "
             "  return key; "
             "};";
    case cccl_type_enum::CCCL_UINT16:
      return "extern \"C\" __device__ void* op(void* key_void) { "
             "  unsigned short* key = reinterpret_cast<unsigned short*>(key_void); "
             "  return key; "
             "};";
    case cccl_type_enum::CCCL_INT32:
      return "extern \"C\" __device__ void* op(void* key_void) { "
             "  int* key = reinterpret_cast<int*>(key_void); "
             "  return key; "
             "};";
    case cccl_type_enum::CCCL_UINT32:
      return "extern \"C\" __device__ void* op(void* key_void) { "
             "  unsigned int* key = reinterpret_cast<unsigned int*>(key_void); "
             "  return key; "
             "};";
    case cccl_type_enum::CCCL_INT64:
      return "extern \"C\" __device__ void* op(void* key_void) { "
             "  long long* key = reinterpret_cast<long long*>(key_void); "
             "  return key; "
             "};";
    case cccl_type_enum::CCCL_UINT64:
      return "extern \"C\" __device__ void* op(void* key_void) { "
             "  unsigned long long* key = reinterpret_cast<unsigned long long*>(key_void); "
             "  return key; "
             "};";
    case cccl_type_enum::CCCL_FLOAT32:
      return "extern \"C\" __device__ void* op(void* key_void) { "
             "  float* key = reinterpret_cast<float*>(key_void); "
             "  return key; "
             "};";
    case cccl_type_enum::CCCL_FLOAT64:
      return "extern \"C\" __device__ void* op(void* key_void) { "
             "  double* key = reinterpret_cast<double*>(key_void); "
             "  return key; "
             "};";
    case cccl_type_enum::CCCL_FLOAT16:
      return "#include <cuda_fp16.h>\n"
             "extern \"C\" __device__ void* op(void* key_void) { "
             "  __half* key = reinterpret_cast<__half*>(key_void); "
             "  return key; "
             "};";

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
#if _CCCL_HAS_NVFP16()
    case cccl_type_enum::CCCL_FLOAT16:
      return "__half";
#endif
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

  pointer_t(std::size_t num_items)
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
  cccl_op_code_type code_type = CCCL_OP_LTOIR; // Default to LTO-IR for backward compatibility

  operation_t() = default;

  operation_t(std::string_view op_name, std::string_view op_code, cccl_op_code_type op_code_type = CCCL_OP_LTOIR)
      : name(op_name)
      , code(op_code)
      , code_type(op_code_type)
  {}

  operator cccl_op_t()
  {
    cccl_op_t op;
    op.type      = cccl_op_kind_t::CCCL_STATELESS;
    op.name      = name.c_str();
    op.code      = code.c_str();
    op.code_size = code.size();
    op.code_type = code_type;
    op.size      = 1;
    op.alignment = 1;
    op.state     = nullptr;
    return op;
  }
};

template <class OpT>
struct stateful_operation_t
{
  OpT op_state;
  std::string name;
  std::string code;

  stateful_operation_t(const OpT& state, std::string_view op_name, std::string_view op_code)
      : op_state(state)
      , name(op_name)
      , code(op_code)
  {}

  operator cccl_op_t()
  {
    cccl_op_t op;
    op.type      = cccl_op_kind_t::CCCL_STATEFUL;
    op.size      = sizeof(OpT);
    op.alignment = alignof(OpT);
    op.state     = &op_state;
    op.name      = name.c_str();
    op.code      = code.c_str();
    op.code_size = code.size();
    op.code_type = CCCL_OP_LTOIR; // Stateful operations always use LTO-IR
    return op;
  }
};

inline operation_t make_operation(std::string_view name, const std::string& code)
{
  return operation_t{name, compile(code), CCCL_OP_LTOIR};
}

inline operation_t make_cpp_operation(std::string_view name, const std::string& cpp_code)
{
  return operation_t{name, cpp_code, CCCL_OP_CPP_SOURCE};
}

template <class OpT>
stateful_operation_t<OpT> make_operation(std::string_view name, const std::string& code, OpT op)
{
  return {op, name, compile(code)};
}

static cccl_op_t make_well_known_unary_operation()
{
  return {cccl_op_kind_t::CCCL_NEGATE, "", "", 0, CCCL_OP_LTOIR, 1, 1, nullptr};
}

static cccl_op_t make_well_known_binary_operation()
{
  return {cccl_op_kind_t::CCCL_PLUS, "", "", 0, CCCL_OP_LTOIR, 1, 1, nullptr};
}

static cccl_op_t make_well_known_binary_predicate()
{
  return {cccl_op_kind_t::CCCL_LESS, "", "", 0, CCCL_OP_LTOIR, 1, 1, nullptr};
}

static cccl_op_t make_well_known_unique_binary_predicate()
{
  return {cccl_op_kind_t::CCCL_EQUAL_TO, "", "", 0, CCCL_OP_LTOIR, 1, 1, nullptr};
}

template <class ValueT, class StateT>
struct iterator_t
{
  StateT state;
  std::string state_name;
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

template <typename BaseIteratorStateTy>
struct stateless_transform_it_state
{
  using BaseIteratorStateT = BaseIteratorStateTy;

  BaseIteratorStateTy base_it_state;
};

template <typename BaseIteratorStateTy, typename FunctorStateTy>
struct stateful_transform_it_state
{
  using BaseIteratorStateT = BaseIteratorStateTy;
  using FunctorStateT      = FunctorStateTy;

  BaseIteratorStateTy base_it_state;
  FunctorStateTy functor_state;
};

struct name_source_t
{
  std::string_view name;
  std::string_view def_src;
};

template <class ValueT, class StateT>
iterator_t<ValueT, StateT> make_iterator(name_source_t state, operation_t advance, operation_t dereference)
{
  iterator_t<ValueT, StateT> it;
  it.state_name                = state.name;
  const std::string& state_src = std::string{state.def_src};
  it.advance                   = make_operation(advance.name, state_src + advance.code);
  it.dereference               = make_operation(dereference.name, state_src + dereference.code);
  return it;
}

inline std::tuple<std::string, std::string, std::string> make_random_access_iterator_sources(
  iterator_kind kind,
  std::string_view value_type,
  std::string_view iterator_state_name,
  std::string_view advance_fn_name,
  std::string_view dereference_fn_name,
  std::string_view transform = "")
{
  std::string state_def_src      = std::format("struct {0} {{ {1}* data; }};\n", iterator_state_name, value_type);
  std::string advance_fn_def_src = std::format(
    "extern \"C\" __device__ void {0}({1}* state, unsigned long long offset) {{\n"
    "  state->data += offset;\n"
    "}}",
    advance_fn_name,
    iterator_state_name);

  std::string dereference_fn_def_src;
  if (kind == iterator_kind::INPUT)
  {
    dereference_fn_def_src = std::format(
      "extern \"C\" __device__ void {0}({1}* state, {2}* result) {{\n"
      "  *result = (*state->data){3};\n"
      "}}",
      dereference_fn_name,
      iterator_state_name,
      value_type,
      transform);
  }
  else
  {
    dereference_fn_def_src = std::format(
      "extern \"C\" __device__ void {0}({1}* state, {2} x) {{\n"
      "  *state->data = x{3};\n"
      "}}",
      dereference_fn_name,
      iterator_state_name,
      value_type,
      transform);
  }

  return std::make_tuple(state_def_src, advance_fn_def_src, dereference_fn_def_src);
}

template <class ValueT>
iterator_t<ValueT, random_access_iterator_state_t<ValueT>> make_random_access_iterator(
  iterator_kind kind, std::string_view value_type, std::string prefix = "", std::string transform = "")
{
  std::string iterator_state_name = std::format("{0}state_t", prefix);
  std::string advance_fn_name     = std::format("{0}advance", prefix);
  std::string dereference_fn_name = std::format("{0}dereference", prefix);

  const auto& [iterator_state_def_src, advance_fn_def_src, dereference_fn_def_src] =
    make_random_access_iterator_sources(
      kind, value_type, iterator_state_name, advance_fn_name, dereference_fn_name, transform);

  name_source_t iterator_state = {iterator_state_name, iterator_state_def_src};
  operation_t advance          = {advance_fn_name, advance_fn_def_src};
  operation_t dereference      = {dereference_fn_name, dereference_fn_def_src};

  return make_iterator<ValueT, random_access_iterator_state_t<ValueT>>(iterator_state, advance, dereference);
}

inline std::tuple<std::string, std::string, std::string> make_counting_iterator_sources(
  std::string_view value_type,
  std::string_view iterator_state_name,
  std::string_view advance_fn_name,
  std::string_view dereference_fn_name)
{
  std::string iterator_state_def_src = std::format("struct {0} {{ {1} value; }};\n", iterator_state_name, value_type);
  std::string advance_fn_def_src     = std::format(
    "extern \"C\" __device__ void {0}({1}* state, unsigned long long offset) {{\n"
        "  state->value += offset;\n"
        "}}",
    advance_fn_name,
    iterator_state_name);

  std::string dereference_fn_def_src = std::format(
    "extern \"C\" __device__ void {0}({1}* state, {2}* result) {{ \n"
    "  *result = state->value;\n"
    "}}",
    dereference_fn_name,
    iterator_state_name,
    value_type);

  return std::make_tuple(iterator_state_def_src, advance_fn_def_src, dereference_fn_def_src);
}

template <class ValueT>
iterator_t<ValueT, counting_iterator_state_t<ValueT>>
make_counting_iterator(std::string_view value_type, std::string_view prefix = "")
{
  std::string iterator_state_name = std::format("{0}state_t", prefix);
  std::string advance_fn_name     = std::format("{0}advance", prefix);
  std::string dereference_fn_name = std::format("{0}dereference", prefix);

  const auto& [iterator_state_src, advance_fn_def_src, dereference_fn_def_src] =
    make_counting_iterator_sources(value_type, iterator_state_name, advance_fn_name, dereference_fn_name);

  name_source_t iterator_state = {iterator_state_name, iterator_state_src};
  operation_t advance          = {advance_fn_name, advance_fn_def_src};
  operation_t dereference      = {dereference_fn_name, dereference_fn_def_src};

  return make_iterator<ValueT, counting_iterator_state_t<ValueT>>(iterator_state, advance, dereference);
}

inline std::tuple<std::string, std::string, std::string> make_constant_iterator_sources(
  std::string_view value_type,
  std::string_view iterator_state_name,
  std::string_view advance_fn_name,
  std::string_view dereference_fn_name)
{
  std::string iterator_state_src = std::format("struct {0} {{ {1} value; }};\n", iterator_state_name, value_type);
  std::string advance_fn_src     = std::format(
    "extern \"C\" __device__ void {0}({1}* state, unsigned long long offset) {{ }}",
    advance_fn_name,
    iterator_state_name);
  std::string dereference_fn_src = std::format(
    "extern \"C\" __device__ void {0}({1}* state, {2}* result) {{ \n"
    "  *result = state->value;\n"
    "}}",
    dereference_fn_name,
    iterator_state_name,
    value_type);

  return std::make_tuple(iterator_state_src, advance_fn_src, dereference_fn_src);
}

template <class ValueT>
iterator_t<ValueT, constant_iterator_state_t<ValueT>>
make_constant_iterator(std::string_view value_type, std::string_view prefix = "")
{
  std::string iterator_state_name = std::format("{0}struct_t", prefix);
  std::string advance_fn_name     = std::format("{0}advance", prefix);
  std::string dereference_fn_name = std::format("{0}dereference", prefix);

  const auto& [iterator_state_src, advance_fn_src, dereference_fn_src] =
    make_constant_iterator_sources(value_type, iterator_state_name, advance_fn_name, dereference_fn_name);

  name_source_t iterator_state = {iterator_state_name, iterator_state_src};
  operation_t advance          = {advance_fn_name, advance_fn_src};
  operation_t dereference      = {dereference_fn_name, dereference_fn_src};

  return make_iterator<ValueT, constant_iterator_state_t<ValueT>>(iterator_state, advance, dereference);
}

inline std::tuple<std::string, std::string, std::string> make_reverse_iterator_sources(
  iterator_kind kind,
  std::string_view value_type,
  std::string_view iterator_state_name,
  std::string_view advance_fn_name,
  std::string_view dereference_fn_name,
  std::string_view transform = "")
{
  std::string iterator_state_src = std::format("struct {0} {{ {1}* data; }};\n", iterator_state_name, value_type);
  std::string advance_fn_src     = std::format(
    "extern \"C\" __device__ void {0}({1}* state, unsigned long long offset) {{\n"
        "  state->data -= offset;\n"
        "}}",
    advance_fn_name,
    iterator_state_name);
  std::string dereference_fn_src;
  if (kind == iterator_kind::INPUT)
  {
    dereference_fn_src = std::format(
      "extern \"C\" __device__ void {0}({1}* state, {2}* result) {{\n"
      "  *result = (*state->data){3};\n"
      "}}",
      dereference_fn_name,
      iterator_state_name,
      value_type,
      transform);
  }
  else
  {
    dereference_fn_src = std::format(
      "extern \"C\" __device__ void {0}({1}* state, {2} x) {{\n"
      "  *state->data = x{3};\n"
      "}}",
      dereference_fn_name,
      iterator_state_name,
      value_type,
      transform);
  }

  return std::make_tuple(iterator_state_src, advance_fn_src, dereference_fn_src);
}

template <class ValueT>
iterator_t<ValueT, random_access_iterator_state_t<ValueT>> make_reverse_iterator(
  iterator_kind kind, std::string_view value_type, std::string_view prefix = "", std::string_view transform = "")
{
  std::string iterator_state_name = std::format("{0}struct_t", prefix);
  std::string advance_fn_name     = std::format("{0}advance", prefix);
  std::string dereference_fn_name = std::format("{0}dereference", prefix);

  const auto& [iterator_state_src, advance_fn_src, dereference_fn_src] = make_reverse_iterator_sources(
    kind, value_type, iterator_state_name, advance_fn_name, dereference_fn_name, transform);

  name_source_t iterator_state = {iterator_state_name, iterator_state_src};
  operation_t advance          = {advance_fn_name, advance_fn_src};
  operation_t dereference      = {dereference_fn_name, dereference_fn_src};

  return make_iterator<ValueT, random_access_iterator_state_t<ValueT>>(iterator_state, advance, dereference);
}

inline std::tuple<std::string, std::string, std::string> make_stateful_transform_input_iterator_sources(
  std::string_view transform_it_state_name,
  std::string_view transform_it_advance_fn_name,
  std::string_view transform_it_dereference_fn_name,
  std::string_view transformed_value_type,
  std::string_view base_value_type,
  name_source_t base_it_state,
  name_source_t base_it_advance_fn,
  name_source_t base_it_dereference_fn,
  name_source_t transform_state,
  name_source_t transform_op)
{
  static constexpr std::string_view transform_it_state_src_tmpl = R"XXX(
/* Define state of stateful transform operation */
{3}
/* Define state of base iterator over whose values transformation is applied */
{4}
struct {0} {{
  {1} base_it_state;
  {2} functor_state;
}};
)XXX";

  const std::string transform_it_state_src = std::format(
    transform_it_state_src_tmpl,
    /* 0 */ transform_it_state_name,
    /* 1 */ base_it_state.name,
    /* 2 */ transform_state.name,
    /* 3 */ transform_state.def_src,
    /* 4 */ base_it_state.def_src);

  static constexpr std::string_view transform_it_advance_fn_src_tmpl = R"XXX(
{3}
extern "C" __device__ void {0}({1} *transform_it_state, unsigned long long offset) {{
    {2}(&(transform_it_state->base_it_state), offset);
}}
)XXX";

  const std::string transform_it_advance_fn_src = std::format(
    transform_it_advance_fn_src_tmpl,
    /* 0 */ transform_it_advance_fn_name,
    /* 1 */ transform_it_state_name,
    /* 2 */ base_it_advance_fn.name,
    /* 3 */ base_it_advance_fn.def_src);

  static constexpr std::string_view transform_it_dereference_fn_src_tmpl = R"XXX(
{5}
{6}
extern "C" __device__ void {0}({1} *transform_it_state, {2}* result) {{
    {7} base_result;
    {4}(&(transform_it_state->base_it_state), &base_result);
    *result = {3}(
        &(transform_it_state->functor_state),
        base_result
    );
}}
)XXX";

  const std::string transform_it_dereference_fn_src = std::format(
    transform_it_dereference_fn_src_tmpl,
    /* 0 */ transform_it_dereference_fn_name /* name of transform's deref function */,
    /* 1 */ transform_it_state_name /* name of transform's state*/,
    /* 2 */ transformed_value_type /* function return type name */,
    /* 3 */ transform_op.name /* transformation functor function name */,
    /* 4 */ base_it_dereference_fn.name /* deref function of base iterator */,
    /* 5 */ base_it_dereference_fn.def_src,
    /* 6 */ transform_op.def_src,
    /* 7 */ base_value_type);

  return std::make_tuple(transform_it_state_src, transform_it_advance_fn_src, transform_it_dereference_fn_src);
}

template <typename ValueT, typename BaseIteratorStateT, typename TransformerStateT>
auto make_stateful_transform_input_iterator(
  std::string_view transformed_value_type,
  std::string_view base_value_type,
  name_source_t base_it_state,
  name_source_t base_it_advance_fn,
  name_source_t base_it_dereference_fn,
  name_source_t transform_state,
  name_source_t transform_op)
{
  static constexpr std::string_view transform_it_state_name          = "stateful_transform_iterator_state_t";
  static constexpr std::string_view transform_it_advance_fn_name     = "advance_stateful_transform_it";
  static constexpr std::string_view transform_it_dereference_fn_name = "dereference_stateful_transform_it";

  const auto& [transform_it_state_src, transform_it_advance_fn_src, transform_it_dereference_fn_src] =
    make_stateful_transform_input_iterator_sources(
      transform_it_state_name,
      transform_it_advance_fn_name,
      transform_it_dereference_fn_name,
      transformed_value_type,
      base_value_type,
      base_it_state,
      base_it_advance_fn,
      base_it_dereference_fn,
      transform_state,
      transform_op);

  using HostTransformStateT = stateful_transform_it_state<BaseIteratorStateT, TransformerStateT>;
  auto transform_it         = make_iterator<ValueT, HostTransformStateT>(
    {transform_it_state_name, transform_it_state_src},
    {transform_it_advance_fn_name, transform_it_advance_fn_src},
    {transform_it_dereference_fn_name, transform_it_dereference_fn_src});

  return transform_it;
}

/*! @brief Generate source code with definitions for state of transformed iterator and functions to operator on it */
inline std::tuple<std::string, std::string, std::string> make_stateless_transform_input_iterator_sources(
  std::string_view transform_it_state_name,
  std::string_view transform_it_advance_fn_name,
  std::string_view transform_it_dereference_fn_name,
  std::string_view transformed_value_type,
  std::string_view base_value_type,
  name_source_t base_it_state,
  name_source_t base_it_advance_fn,
  name_source_t base_it_dereference_fn,
  name_source_t transform_op)
{
  static constexpr std::string_view transform_it_state_src_tmpl = R"XXX(
/* Define state of base iterator over whose values transformation is applied */
{2}
struct {0} {{
  {1} base_it_state;
}};
)XXX";

  const std::string transform_it_state_src = std::format(
    transform_it_state_src_tmpl,
    /* 0 */ transform_it_state_name,
    /* 1 */ base_it_state.name,
    /* 2 */ base_it_state.def_src);

  static constexpr std::string_view transform_it_advance_fn_src_tmpl = R"XXX(
{3}
extern "C" __device__ void {0}({1} *transform_it_state, unsigned long long offset) {{
    {2}(&(transform_it_state->base_it_state), offset);
}}
)XXX";

  const std::string transform_it_advance_fn_src = std::format(
    transform_it_advance_fn_src_tmpl,
    /* 0 */ transform_it_advance_fn_name,
    /* 1 */ transform_it_state_name,
    /* 2 */ base_it_advance_fn.name,
    /* 3 */ base_it_advance_fn.def_src);

  static constexpr std::string_view transform_it_dereference_fn_src_tmpl = R"XXX(
{5}
{6}
extern "C" __device__ void {0}({1} *transform_it_state, {2}* result) {{
    {7} base_result;
    {4}(&(transform_it_state->base_it_state), &base_result);
    *result = {3}(base_result);
}}
)XXX";

  const std::string transform_it_dereference_fn_src = std::format(
    transform_it_dereference_fn_src_tmpl,
    /* 0 */ transform_it_dereference_fn_name /* name of transform's deref function */,
    /* 1 */ transform_it_state_name /* name of transform's state*/,
    /* 2 */ transformed_value_type /* function return type name */,
    /* 3 */ transform_op.name /* transformation functor function name */,
    /* 4 */ base_it_dereference_fn.name /* deref function of base iterator */,
    /* 5 */ base_it_dereference_fn.def_src,
    /* 6 */ transform_op.def_src,
    /* 7 */ base_value_type);

  return std::make_tuple(transform_it_state_src, transform_it_advance_fn_src, transform_it_dereference_fn_src);
}

template <typename ValueT, typename BaseIteratorStateT>
auto make_stateless_transform_input_iterator(
  std::string_view transformed_value_type,
  std::string_view base_value_type,
  name_source_t base_it_state,
  name_source_t base_it_advance_fn,
  name_source_t base_it_dereference_fn,
  name_source_t transform_op)
{
  static constexpr std::string_view transform_it_state_name      = "stateless_transform_iterator_state_t";
  static constexpr std::string_view transform_it_advance_fn_name = "advance_stateless_transform_it";
  static constexpr std::string_view transform_it_deref_fn_name   = "dereference_stateless_transform_it";

  const auto& [transform_it_state_src, transform_it_advance_fn_src, transform_it_deref_fn_src] =
    make_stateless_transform_input_iterator_sources(
      transform_it_state_name,
      transform_it_advance_fn_name,
      transform_it_deref_fn_name,
      transformed_value_type,
      base_value_type,
      base_it_state,
      base_it_advance_fn,
      base_it_dereference_fn,
      transform_op);

  using HostTransformStateT = stateless_transform_it_state<BaseIteratorStateT>;
  auto transform_it         = make_iterator<ValueT, HostTransformStateT>(
    {transform_it_state_name, transform_it_state_src},
    {transform_it_advance_fn_name, transform_it_advance_fn_src},
    {transform_it_deref_fn_name, transform_it_deref_fn_src});

  return transform_it;
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
