# cuda-cccl

`cuda-cccl` is the aggregate Python metapackage for the CUDA Core Compute
Libraries. It contains no importable modules. Installing it brings in the
same-version `cuda-compute` distribution. The headers needed by
`cuda.compute` are private data inside that wheel.

Install the aggregate package from PyPI with the CUDA extra matching your
environment:

```bash
pip install cuda-cccl[cu13]  # CUDA 13.x with a pip-installed toolkit
pip install cuda-cccl[cu12]  # CUDA 12.x with a pip-installed toolkit
```

The `sysctk12`, `sysctk13`, `minimal-*`, `test-*`, and `bench-*` extras are
forwarded to `cuda-compute` as well.

## Upgrading from the former monolithic wheel

The former `cuda-cccl` wheel owned the `cuda.compute` files that are now owned
by `cuda-compute`, along with compatibility modules under `cuda.cccl` that have
been removed. Before the first upgrade to this split layout, uninstall the old
wheel in a separate command so pip cannot remove the new owner's files while
processing the old wheel's RECORD:

```bash
python -m pip uninstall -y cuda-cccl
python -m pip install "cuda-cccl[cu13]"  # or the extra for your environment
```

This is a one-time migration. Subsequent upgrades within the two-wheel layout
can use the normal `pip install --upgrade` flow.

## Documentation

See the [CCCL Python documentation](https://nvidia.github.io/cccl/unstable/python)
for complete installation and API guidance.
