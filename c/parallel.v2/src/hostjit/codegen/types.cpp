#include <format>

#include <hostjit/codegen/types.hpp>

namespace hostjit::codegen
{
std::string get_type_name(cccl_type_enum type)
{
  switch (type)
  {
    case CCCL_INT8:
      return "char";
    case CCCL_INT16:
      return "short";
    case CCCL_INT32:
      return "int";
    case CCCL_INT64:
      return "long long";
    case CCCL_UINT8:
      return "unsigned char";
    case CCCL_UINT16:
      return "unsigned short";
    case CCCL_UINT32:
      return "unsigned int";
    case CCCL_UINT64:
      return "unsigned long long";
    case CCCL_FLOAT16:
      return "__half";
    case CCCL_FLOAT32:
      return "float";
    case CCCL_FLOAT64:
      return "double";
    case CCCL_BOOLEAN:
      return "bool";
    default:
      return "";
  }
}

std::string make_storage_type(const char* name, size_t size, size_t alignment)
{
  return std::format(
    "struct __align__({}) {} {{\n"
    "  char data[{}];\n"
    "}};\n",
    alignment,
    name,
    size);
}

std::string resolve_type(cccl_type_info info, const char* fallback_alias, std::string& out_preamble)
{
  auto name = get_type_name(info.type);
  if (!name.empty())
  {
    return name;
  }
  // Custom type: emit storage struct definition, return alias
  out_preamble += make_storage_type(fallback_alias, info.size, info.alignment);
  return fallback_alias;
}
} // namespace hostjit::codegen
