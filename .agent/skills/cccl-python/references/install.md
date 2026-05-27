# Install methods — `cuda-cccl`

The `cuda-cccl` Python package ships through three channels: PyPI, conda-forge, and from
source. Pick by environment.

## Requirements

- Python 3.10, 3.11, 3.12, or 3.13.
- CUDA Toolkit 12.x or 13.x.
- NVIDIA GPU with compute capability 6.0 or newer.
- Base Python dependencies: `numba>=0.60.0`, `numpy`, `cuda-pathfinder>=1.2.3`, `cuda-core`,
  `typing_extensions`.

The `[cu12]` / `[cu13]` install extras pull `cuda-bindings`, `cuda-toolkit`, and
`numba-cuda` for the matching CTK major.

## PyPI

```bash
pip install cuda-cccl[cu13]   # or [cu12] for CTK 12.x
```

The `[cu13]` / `[cu12]` extras attach the matching CUDA bindings and Numba support.

## conda-forge

```bash
conda install -c conda-forge cccl-python
```

## From source

```bash
git clone https://github.com/NVIDIA/cccl.git
cd cccl/python/cuda_cccl
pip install -e .[test-cu13]   # or [test-cu12] for CTK 12.x
```

The `[test-*]` extras add test dependencies on top of the base install set; use these in
dev environments. For pure runtime, use `.[cu12]` or `.[cu13]`.

## Verification

Quick sanity check after install:

```bash
python -c "import cuda.compute, cuda.coop._experimental, cuda.cccl.headers; print('ok')"
```

Should print `ok` without errors and without warnings about missing CUDA libraries.
