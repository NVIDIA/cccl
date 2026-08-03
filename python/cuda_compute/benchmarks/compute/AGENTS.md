# CCCL Python cuda.compute benchmarks

This directory contains the code for the Python cuda.compute benchmarks.
They are migrated from the original C++ benchmarks and they should match the C++ implementations
as closely as possible.

The original C++ benchmarks are available in this repository in: ../../../../cub/benchmarks/bench/
We follow the same directory structure and naming conventions converting to Python were appropriate.

The code for cuda.compute is in this repository under:
`../../../../python/cuda_compute/cuda/compute/`. Look into this directory when
searching for existing APIs in Python.

The same-version CCCL headers are provided by the sibling
`../../../../python/cccl_headers/` project. Local benchmark environments must
install that project before or alongside `cuda-compute` so its exact dependency
resolves from the checkout.

The benchmarks use nvbench to run the benchmarks and report the results.
