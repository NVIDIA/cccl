//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/print>

// template<class... Args>
//   void print(format_string<Args...> fmt, Args&&... args);

#include <cuda/std/__print_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#if _CCCL_HOSTED()
#  include <cstdio>
#  include <string>
#endif // _CCCL_HOSTED()

#include "test_macros.h"

#if _CCCL_HOSTED()
const auto tmp_file_name = std::tmpnam(nullptr);

void init()
{
  // Redirect stdout to a temporary file.
  assert(tmp_file_name != nullptr);
  assert(std::freopen(tmp_file_name, "w+", stdout) == stdout);
}

std::string read_stdout()
{
  assert(std::fseek(stdout, 0, SEEK_END) == 0);

  auto size = std::ftell(stdout);
  assert(size >= 0);

  std::string ret(size, '\0');
  std::rewind(stdout);
  assert(std::fread(ret.data(), 1, size, stdout) == static_cast<cuda::std::size_t>(size));

  return ret;
}

void deinit()
{
  // Close and remove the temporary file.
  assert(std::fclose(stdout) == 0);
  assert(std::remove(tmp_file_name) == 0);
}

template <class... Args>
bool check(cuda::std::string_view ref, cuda::std::format_string<Args...> fmt, Args&&... args)
{
  init();

  cuda::std::print(cuda::std::forward<decltype(fmt)>(fmt), cuda::std::forward<decltype(args)>(args)...);
  auto contents = read_stdout();
  assert(contents == ref);

  deinit();

  return true;
}

void test_host()
{
  // *** Test escaping  ***

  assert(check("{", "{{"));
  assert(check("{:^}", "{{:^}}"));
  assert(check("{: ^}", "{{:{}^}}", ' '));
  assert(check("{:{}^}", "{{:{{}}^}}"));
  assert(check("{:{ }^}", "{{:{{{}}}^}}", ' '));

  // *** Test argument ID ***
  assert(check("hello false true", "hello {0:} {1:}", false, true));
  assert(check("hello true false", "hello {1:} {0:}", false, true));

  // *** Test many arguments ***
  assert(check(
    "1234567890\t1234567890",
    "{}{}{}{}{}{}{}{}{}{}\t{}{}{}{}{}{}{}{}{}{}",
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    0));

  // *** Test embedded NUL character ***
  using namespace cuda::std::literals;
  assert(check("hello\0world"sv, "hello{}{}", '\0', "world"));
  assert(check("hello\0world"sv, "hello\0{}"sv, "world"));
  assert(check("hello\0world"sv, "hello{}", "\0world"sv));
}
#endif // _CCCL_HOSTED()

__global__ void test_device_kernel()
{
  cuda::std::print("{{");
  cuda::std::print("{{:^}}");
  cuda::std::print("{{:{}^}}", ' ');
  cuda::std::print("{{:{{}}^}}");
  cuda::std::print("{{:{{{}}}^}}", ' ');

  // *** Test argument ID ***
  cuda::std::print("hello {0:} {1:}", false, true);
  cuda::std::print("hello {1:} {0:}", false, true);

  // *** Test many arguments ***
  cuda::std::print(
    "{}{}{}{}{}{}{}{}{}{}\t{}{}{}{}{}{}{}{}{}{}", 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0);
}

#if _CCCL_HOSTED()
void test_device()
{
  // On device, we output everything at once an check the final string.
  constexpr cuda::std::string_view ref_output{
    "{"
    "{:^}"
    "{: ^}"
    "{:{}^}"
    "{:{ }^}"
    "hello false true"
    "hello true false"
    "1234567890\t1234567890"};

  init();

  test_device_kernel<<<1, 1>>>();
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaDeviceSynchronize() == cudaSuccess);

  auto contents = read_stdout();
  assert(contents == ref_output);

  deinit();
}

void test_body()
{
  test_host();
  test_device();
}
#endif // _CCCL_HOSTED()

int main(int, char**)
{
  static_assert(cuda::std::is_same_v<void, decltype(cuda::std::print(""))>);
  static_assert(!noexcept(cuda::std::print("")));

  NV_IF_TARGET(NV_IS_HOST, (test_body();))
  return 0;
}
