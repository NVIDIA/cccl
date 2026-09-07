# Provider qualification

This tree holds broad evidence that is intentionally outside the default pull
request suite. It currently contains the Numba-CUDA-MLIR final-link
qualification suites under `numba_mlir/`, which prove that the shared
public-CUB provider wrappers disappear from fully linked kernels.

Run the qualification selection serially:

```bash
python -m pytest -n 0 -m qualification tests/providers/qualification
```

Each qualification run must record the source commit, imported backend origins
and versions, CCCL headers, compiler/toolkit, GPU architecture, and retained
artifacts. Qualification results extend confidence for a specific stack; they
do not silently promote an unsupported manifest entry.
