#pragma once

// Use fully qualified names to avoid namespace pollution
#include <cuda/std/span>
#include <cstddef>

namespace cuda::experimental {

// Import only span from CUDA Standard Library using fully qualified names
template<typename T, ::std::size_t Extent = ::cuda::std::dynamic_extent>
using span = ::cuda::std::span<T, Extent>;

constexpr ::std::size_t dynamic_extent = ::cuda::std::dynamic_extent;

} // namespace cuda::experimental