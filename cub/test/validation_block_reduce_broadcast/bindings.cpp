// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <torch/extension.h>

at::Tensor normalize(const at::Tensor& input, bool first_thread, bool broadcast);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
  module.def("normalize",
             &normalize,
             pybind11::arg("input"),
             pybind11::arg("first_thread") = false,
             pybind11::arg("broadcast")    = true);
}
