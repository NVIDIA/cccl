# cuda.compute Benchmarks

Compare Python `cuda.compute` performance against C++ CUB implementations.

## Setup

```bash
conda env create -f environment.yml
conda activate cuda-compute-bench
```

Install `cuda.compute`:

```bash
conda install -c conda-forge cccl-python
```

### Build C++ Benchmarks

Build CUB benchmarks using the CI script (one-time, ~13 minutes):

```bash
cd /path/to/cccl
./ci/build_cub.sh -arch 89  # Use your GPU arch (89=RTX 4090, 80=A100, 90=H100)
```

Binaries are built to: `build/cub/bin/`

## Run Benchmarks

```bash
# Run both C++ and Python (default)
./run_benchmarks.sh -b fill -d 0

# Run only C++
./run_benchmarks.sh -b fill --cpp

# Run only Python
./run_benchmarks.sh -b fill --py

# Show help
./run_benchmarks.sh --help
```

## Compare Results

```bash
python analysis/python_vs_cpp_summary.py -b fill
```

## Manual Usage

### List benchmark configurations

```bash
# Python
python nvbench_fill.py --list

# C++
/path/to/cccl/build/cub/bin/cub.bench.transform.fill.base --list
```

### Run with custom options

```bash
# Python - specific type and size
python nvbench_fill.py --axis "T=I32" --axis "Elements=20" --devices 0

# C++ - save JSON
/path/to/cccl/build/cub/bin/cub.bench.transform.fill.base \
  --json results/fill_cpp.json \
  --devices 0
```

### Compare manually

```bash
python analysis/python_vs_cpp_summary.py \
  results/fill_py.json \
  results/fill_cpp.json \
  --device 0
```
