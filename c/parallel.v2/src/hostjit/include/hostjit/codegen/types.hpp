#pragma once

#include <string>

#include <cccl/c/types.h>

namespace hostjit::codegen
{
// Maps cccl_type_enum to plain C/C++ type names (e.g., "int", "float").
// Returns "" for CCCL_STORAGE (caller must handle custom types).
std::string get_type_name(cccl_type_enum type);

// Generates an aligned storage struct definition.
// Example: "struct __align__(8) my_storage_t {\n  char data[16];\n};\n"
std::string make_storage_type(const char* name, size_t size, size_t alignment);

// Returns the C++ type name for a cccl_type_info.
// For known types, returns the type name directly.
// For CCCL_STORAGE, emits a storage struct definition into `out_preamble`
// and returns `fallback_alias`.
std::string resolve_type(cccl_type_info info, const char* fallback_alias, std::string& out_preamble);
} // namespace hostjit::codegen
