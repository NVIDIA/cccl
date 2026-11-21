// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <new>
#include <string>
#include <vector>

#include <nvrtc.h>

#include "provider/runtime/runtime.h"

namespace
{
enum class failure
{
  none,
  create,
  compile,
  size,
  data,
  output_allocation,
  log_allocation,
};

failure selected_failure  = failure::none;
bool fail_next_allocation = false;
int program_storage       = 0;
int create_count          = 0;
int destroy_count         = 0;
constexpr char output[]   = ".global .u64 cuda_coop_tile_resource_sizes[3] = {0x43554f5052455355, 2, 64};";

nvrtcResult get_size(size_t* size)
{
  *size                = sizeof(output);
  fail_next_allocation = selected_failure == failure::output_allocation;
  return selected_failure == failure::size ? NVRTC_ERROR_INTERNAL_ERROR : NVRTC_SUCCESS;
}

nvrtcResult get_data(char* data)
{
  std::memcpy(data, output, sizeof(output));
  return selected_failure == failure::data ? NVRTC_ERROR_INTERNAL_ERROR : NVRTC_SUCCESS;
}
} // namespace

// Fail the first allocation after the fake NVRTC reports the output or log
// size. This exercises the real vector/string allocation after program creation.
void* operator new(size_t size)
{
  if (fail_next_allocation)
  {
    fail_next_allocation = false;
    throw std::bad_alloc{};
  }
  if (void* const result = std::malloc(size == 0 ? 1 : size))
  {
    return result;
  }
  throw std::bad_alloc{};
}

void operator delete(void* ptr) noexcept
{
  std::free(ptr);
}

void operator delete(void* ptr, size_t) noexcept
{
  std::free(ptr);
}

extern "C" nvrtcResult
nvrtcCreateProgram(nvrtcProgram* program, const char*, const char*, int, const char* const*, const char* const*)
{
  if (selected_failure == failure::create)
  {
    return NVRTC_ERROR_OUT_OF_MEMORY;
  }
  *program = reinterpret_cast<nvrtcProgram>(&program_storage);
  ++create_count;
  return NVRTC_SUCCESS;
}

extern "C" nvrtcResult nvrtcDestroyProgram(nvrtcProgram* program)
{
  assert(*program == reinterpret_cast<nvrtcProgram>(&program_storage));
  *program = nullptr;
  ++destroy_count;
  return NVRTC_SUCCESS;
}

extern "C" nvrtcResult nvrtcCompileProgram(nvrtcProgram, int, const char* const*)
{
  return selected_failure == failure::compile || selected_failure == failure::log_allocation
         ? NVRTC_ERROR_COMPILATION
         : NVRTC_SUCCESS;
}

extern "C" nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram, size_t* size)
{
  *size                = sizeof(output);
  fail_next_allocation = selected_failure == failure::log_allocation;
  return NVRTC_SUCCESS;
}

extern "C" nvrtcResult nvrtcGetProgramLog(nvrtcProgram, char* log)
{
  std::memcpy(log, output, sizeof(output));
  return NVRTC_SUCCESS;
}

extern "C" nvrtcResult nvrtcGetLTOIRSize(nvrtcProgram, size_t* size)
{
  return get_size(size);
}

extern "C" nvrtcResult nvrtcGetLTOIR(nvrtcProgram, char* data)
{
  return get_data(data);
}

extern "C" nvrtcResult nvrtcGetPTXSize(nvrtcProgram, size_t* size)
{
  return get_size(size);
}

extern "C" nvrtcResult nvrtcGetPTX(nvrtcProgram, char* data)
{
  return get_data(data);
}

namespace cuda::coop::tile::detail::provider
{
void emit_trace_line(const std::string&) {}
} // namespace cuda::coop::tile::detail::provider

int main()
{
  using namespace cuda::coop::tile::detail::provider;
  for (const bool resource_query : {false, true})
  {
    for (const auto test_failure :
         {failure::none,
          failure::create,
          failure::compile,
          failure::size,
          failure::data,
          failure::output_allocation,
          failure::log_allocation})
    {
      selected_failure = test_failure;
      create_count = destroy_count = 0;
      std::vector<char> ltoir;
      std::vector<uint64_t> sizes;
      std::string error;
      bool threw     = false;
      bool succeeded = false;
      try
      {
        succeeded = resource_query
                    ? compile_resource_query_to_sizes("source", TargetArchitecture{90}, {}, 1, &sizes, &error)
                    : compile_source_to_ltoir("source", "entry", TargetArchitecture{90}, 128, 1, {}, &ltoir, &error);
      }
      catch (const std::bad_alloc&)
      {
        threw = true;
      }
      assert(!fail_next_allocation);
      assert(create_count == (test_failure == failure::create ? 0 : 1));
      assert(destroy_count == create_count);
      assert(threw == (test_failure == failure::output_allocation || test_failure == failure::log_allocation));
      assert(succeeded == (test_failure == failure::none));
      if (succeeded)
      {
        assert(error.empty());
        if (resource_query)
        {
          assert(sizes == std::vector<uint64_t>{64});
        }
        else
        {
          assert(ltoir == std::vector<char>(output, output + sizeof(output)));
        }
      }
      else if (!threw)
      {
        assert(error.find("[LP-PROVIDER-INTERNAL-001] NVRTC error:") == 0);
      }
    }
  }
}
