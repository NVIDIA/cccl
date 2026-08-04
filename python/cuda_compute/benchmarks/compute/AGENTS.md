# CCCL Python cuda.compute benchmarks

This directory contains the code for the Python cuda.compute benchmarks.
They are migrated from the original C++ benchmarks and they should match the C++ implementations
as closely as possible.

The original C++ benchmarks are available in this repository in: ../../../../cub/benchmarks/bench/
We follow the same directory structure and naming conventions converting to Python were appropriate.

The code for cuda.compute is in this repository under:
`../../../../python/cuda_compute/cuda/compute/`. Look into this directory when
searching for existing APIs in Python.

Editable `cuda-compute` installs resolve CCCL headers directly from the
canonical `libcudacxx/`, `cub/`, and `thrust/` directories in this checkout.
Do not copy those headers into the Python project.

The benchmarks use nvbench to run the benchmarks and report the results.
