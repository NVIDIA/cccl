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
python run_benchmarks.py -b transform/fill -d 0

# Run only C++
python run_benchmarks.py -b transform/fill --cpp

# Run only Python
python run_benchmarks.py -b transform/fill --py

# Show help
python run_benchmarks.py --help
```

## Compare Results

```bash
python analysis/python_vs_cpp_summary.py -b transform/fill
```

## Manual Usage

### List benchmark configurations

```bash
# Python
python transform/fill.py --list

# C++
/path/to/cccl/build/cub/bin/cub.bench.transform.fill.base --list
```

### Run with custom options

```bash
# Python - specific type and size
python transform/fill.py --axis "T=I32" --axis "Elements[pow2]=20" --devices 0

# C++ - save JSON
/path/to/cccl/build/cub/bin/cub.bench.transform.fill.base \
  --json results/transform/fill_cpp.json \
  --devices 0
```

### Compare manually

```bash
python analysis/python_vs_cpp_summary.py \
  results/transform/fill_py.json \
  results/transform/fill_cpp.json \
  --device 0
```
