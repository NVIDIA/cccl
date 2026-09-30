# CCCL Python cuda.compute benchmarks

This directory contains the code for the Python cuda.compute benchmarks.
They are migrated from the original C++ benchmarks and they should match the C++ implementations
as closely as possible.

The original C++ benchmarks are available in this repository in: ../../../../cub/benchmarks/bench/
We follow the same directory structure and naming conventions converting to Python were appropriate.

The code for cuda.compute is in this repository under: `../../../../python/cuda_cccl/cuda/compute/`. Look into this directory when searching for existing APIs in Python.

The benchmarks use nvbench to run the benchmarks and report the results.
