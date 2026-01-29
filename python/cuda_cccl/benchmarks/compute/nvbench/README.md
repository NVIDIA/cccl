# cuda.compute Benchmarks

## Setup

```bash
conda env create -f environment.yml
conda activate cuda-compute-bench
```

Then install nvbench (from source at the moment):

```bash
cd nvbench/python
rm -rf build  # cleanup just in case
pip install -e .
```

Install cuda-compute, for example:

```bash
conda install -c conda-forge cccl-python
```

### Build C++ Benchmarks

```bash
mkdir -p build && cd build

# Auto-detect GPU (requires GPU present at build time)
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
# Or specify architecture
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80  # Ampere (A100)
# cmake .. -DCMAKE_CUDA_ARCHITECTURES=90  # Hopper (H100)

# Build
cmake --build . -j
```

Binaries will be output to: `./bin`


## Usage

Run a single Python benchmark:

```bash
# List available benchmarks
python nvbench_transform.py --list

# Run a single benchmark, output to stdout
python nvbench_transform.py --benchmark bench_unary_transform_pointer

# Save a JSON output
mkdir -p results
python nvbench_transform.py --benchmark bench_unary_transform_pointer --json results/bench_transform_py.json
```

Run a single C++ benchmark:

```bash
# List available benchmarks
./bin/nvbench_transform_cpp --list

# Run a single benchmark, output to stdout
./bin/nvbench_transform_cpp --benchmark bench_unary_transform_pointer

# Save a JSON output
mkdir -p results
./bin/nvbench_transform_cpp --benchmark bench_unary_transform_pointer --json results/bench_transform_cpp.json
```

Compare results:

```
python analysis/python_vs_cpp_summary.py results/bench_transform_py.json results/bench_transform_cpp.json
```
