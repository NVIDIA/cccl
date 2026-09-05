# `cuda.coop` examples

These programs compile one block, run it on a CUDA device, and check the result
against a host calculation. Run them from `python/cuda_coop` so Python can find
the `examples` package and this source checkout:

```console
python -m examples.numba_mlir.common_block_scan
python -m examples.numba_mlir.qualified_histogram_decode
python -m examples.cutlass.common_block_scan
python -m examples.cutlass.qualified_radix_topk
```

The CUTLASS examples use PyTorch tensors as DLPack-backed device buffers.
PyTorch is an example dependency rather than a dependency of `cuda-coop`.

The two `common_block_scan` programs use the portable `from cuda import coop`
surface. Their kernel bodies differ only in the compiler decorator and launch
syntax. The qualified examples show controls that belong to one backend's
public API, including fixed scratch storage and optional run-length outputs.
