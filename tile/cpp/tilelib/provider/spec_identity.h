// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string>

#include "provider/primitive_spec.h"

namespace cuda::coop::tile::detail::provider
{
std::string codegen_metadata(const ImplementationSpec& request);
std::string semantic_metadata(const ImplementationSpec& request);
std::string semantic_identity(const ImplementationSpec& request);
std::string artifact_identity(const ImplementationSpec& request);
std::string artifact_identity(const ImplementationSpec& request, TargetArchitecture target);
std::string request_payload(const PrimitiveSpec& primitive);
} // namespace cuda::coop::tile::detail::provider
