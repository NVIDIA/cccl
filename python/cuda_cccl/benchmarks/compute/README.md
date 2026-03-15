# cuda.compute Benchmarks

Compare Python `cuda.compute` performance against C++ CUB implementations.

## Setup

This project uses [pixi](https://pixi.sh) to manage environments and dependencies.

Two environments are available:

- **`wheel`** - Uses the released `cuda-cccl` package
- **`source`** - Builds `cuda-cccl` from the local repository

### Build C++ Benchmarks

Build CUB benchmarks using the CI script (one-time, ~13 minutes):

```bash
cd /path/to/cccl
./ci/build_cub.sh -arch 89  # Use your GPU arch (89=RTX 4090, 80=A100, 90=H100)
```

Binaries are built to: `build/cub/bin/`

## Run Benchmarks

### Using pixi tasks

```bash
# Run Python benchmarks (released cuda-cccl)
pixi run -e wheel bench

# Run Python benchmarks (local source build)
pixi run -e source bench

# Run Python benchmarks with reduced parameter set
pixi run -e wheel bench-quick

# Run just one benchmark
pixi run -e wheel bench -b transform/fill

# Run C++ benchmarks
pixi run -e wheel bench-cpp

# Run both Python and C++ benchmarks
pixi run -e wheel bench-all
```

### Using run_benchmarks.py directly

```bash
# Run both C++ and Python (default)
pixi run -e wheel python run_benchmarks.py -b transform/fill -d 0

# Run only C++
pixi run -e wheel python run_benchmarks.py -b transform/fill --cpp

# Run only Python
pixi run -e wheel python run_benchmarks.py -b transform/fill --py

# Show help
pixi run -e wheel python run_benchmarks.py --help
```

To run the benchmarks using the "quick" configuration:

```bash
pixi run -e wheel python run_benchmarks.py --quick
```

## Compare Results

```bash
pixi run -e wheel python analysis/python_vs_cpp_summary.py -b transform/fill
```

## Web Report

A simple page used to visualize a set of results.

- Requires `results/` to be populated with benchmark results.

First generate a manifest:

```bash
pixi run -e wheel python analysis/generate_web_report_manifest.py \
  --results-dir results \
  --output results/manifest.json
```

Build the web report single file app:

```bash
cd analysis/web-report
npm install
npm run build
```

This will output a single file app to `analysis/web-report/dist/` copy it to the `results/` directory and:

```bash
cd results/
python3 -m http.server
```

Now its possible to share the results directory as a zip/tar file.

## Manual Usage

### List benchmark configurations

```bash
# Python
pixi run -e wheel python transform/fill.py --list

# C++
/path/to/cccl/build/cub/bin/cub.bench.transform.fill.base --list
```

### Run with custom options

```bash
# Python - specific type and size
pixi run -e wheel python transform/fill.py --axis "T=I32" --axis "Elements[pow2]=20" --devices 0

# C++ - save JSON
/path/to/cccl/build/cub/bin/cub.bench.transform.fill.base \
  --json results/transform/fill_cpp.json \
  --devices 0
```

### Compare manually

```bash
pixi run -e wheel python analysis/python_vs_cpp_summary.py \
  results/transform/fill_py.json \
  results/transform/fill_cpp.json \
  --device 0
```

## AI commands

These are using the .opencode folder but can be moved to other Agents.

### /migration-status

Generates a report of the migration status for each benchmark in CUB.
