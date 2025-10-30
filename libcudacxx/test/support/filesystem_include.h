#ifndef TEST_SUPPORT_FILESYSTEM_INCLUDE_H
#define TEST_SUPPORT_FILESYSTEM_INCLUDE_H

#include <filesystem>

#include "test_macros.h"

#if defined(_CUDA_STD_VERSION)
namespace fs = std::__fs::filesystem;
#else
namespace fs = std::filesystem;
#endif

#endif
