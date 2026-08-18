# CUDA STF Python Package

[`cuda.stf._experimental`](https://nvidia.github.io/cccl/python/stf.html)
provides Python bindings to **CUDASTF (Sequential Task Flow)**: you define logical
data and submit tasks that read or write that data, and STF infers the
dependencies and orchestrates execution and data movement. It is part of the
[CUDA Core Compute Libraries](https://nvidia.github.io/cccl/cpp.html#cccl-cpp-libraries).

The API is exposed under the `_experimental` subpackage because it is still
evolving and may change without notice. CUDASTF is currently **Linux-only**.

## Installation

Install from PyPI:

```bash
pip install cuda-stf[cu13]  # For CUDA 13.x (pip-installed cuda-toolkit)
pip install cuda-stf[cu12]  # For CUDA 12.x (pip-installed cuda-toolkit)
```

If you already have a CUDA toolkit on your system and do not want pip to
install it, use the `sysctk` variants:

```bash
pip install cuda-stf[sysctk13]  # For CUDA 13.x (system CUDA toolkit)
pip install cuda-stf[sysctk12]  # For CUDA 12.x (system CUDA toolkit)
```

For a smaller install without Numba (when you drive kernels through
`cuda.core` / `cuda.compute` or your own launches), use the `minimal-*`
variants:

```bash
pip install cuda-stf[minimal-cu13]       # pip CUDA toolkit, no Numba
pip install cuda-stf[minimal-sysctk13]   # system CUDA toolkit, no Numba
```

Install `cuda-cccl` as well when using `cuda.compute` with STF or compiling
external C++ code against the cudax headers; it supplies the libcudacxx, CUB,
and Thrust headers.

Feature dependencies are installed separately as needed: `cuda-cccl`
(`cuda.compute` and header discovery), `numba` / `numba-cuda` (Numba interop,
bundled by the non-`minimal` extras), `cupy`, `torch` (PyTorch interop),
`warp-lang` (Warp interop), and `nvmath-python` (cuBLAS/cuSOLVER examples).

### Install from source (Linux only)

```bash
git clone https://github.com/NVIDIA/cccl.git
cd cccl/python/cuda_stf
pip install -e .[test-cu13]  # or .[test-cu12], .[test-sysctk13], .[test-sysctk12]
```

Building from source compiles the native `cccl.c.experimental.stf` / `cudax`
extension, so a C++ toolchain and CMake (`>=3.30`) with Ninja are required in
addition to the CUDA toolkit. The `test-*` extras add `cuda-cccl`, `pytest`,
`pytest-xdist`, and CuPy so the test suite (`pytest tests/`) can run.

**Requirements:** Python 3.10+, CUDA Toolkit 12.x or 13.x, NVIDIA GPU with
Compute Capability 7.5+, Linux.

## Device memory interchange: CAI and DLPack

`DeviceArray` (buffers allocated through a `data_place`, including composite
localized places) implements **both** interchange protocols. They are
complementary — a consumer picks the semantics by construction:

- **CUDA Array Interface** (`__cuda_array_interface__`, CAI v3) *describes*
  the memory and transfers **no ownership**: the `DeviceArray` must outlive
  every borrowed view. This is the zero-copy path used by task arguments,
  Numba, `cuda.compute`, and `torch.as_tensor`.
- **DLPack** (`__dlpack__` / `__dlpack_device__`) *carries ownership*: the
  exported capsule keeps the array alive and the consumer's deleter releases
  it, so e.g. `torch.from_dlpack(arr)` yields a tensor whose storage lifetime
  owns the allocation. Deallocation stays with the `DeviceArray` finalizer —
  one deallocation point regardless of protocol.

```python
stf.machine_init()
arr = stf.DeviceArray((4, 8), np.float32, stf.data_place.device(0))
borrowed = numba.cuda.as_cuda_array(arr)   # CAI: arr must stay alive
owned    = torch.from_dlpack(arr)          # DLPack: the tensor keeps it alive
```

For pytorch-flavored code, an optional convenience attaches the factory
family as a `torch.localized` namespace (an attribute plus a `sys.modules`
entry -- purely additive, nothing about torch's own behavior changes, and
`uninstall()` reverses it):

```python
import torch
import cuda.stf._experimental as stf

stf.interop.pytorch.install()      # adds torch.localized
stf.machine_init()
grid = stf.exec_place_grid.from_devices([0, 1])

w = torch.localized.parameter((4096, 4096), torch.bfloat16, grid,
                              spec=(("blocked", 0), None))
# no spec => the default: blocked along dim 0 (here: batch rows split
# across the grid). Placement granularity is the 2 MiB VMM page, so give
# the split something to work with -- a tensor smaller than one page lands
# on a single place no matter the spec.
x = torch.localized.zeros((8192, 4096), torch.float32, grid)
b = torch.localized.zeros_like(x)  # reuses x's placement verbatim
torch.localized.placement_report(x)  # dry-run: bytes per grid position
```

Compute follows the same model: `torch.localized.map(fn, *tensors)` applies
a map expression (eager, or a stock `torch.compile` artifact — fusion stays
torch's job) once per die, each over a strided view of exactly the die's
elements, forked/joined with events so the whole thing is CUDA-graph
capturable. The iteration split is inferred from the operands' placement
(all localized operands must share one spec; ordinary broadcast scalars
pass whole).
Valid bodies are maps w.r.t. the split axes: pointwise always, dim-wise ops
along unsplit dims (softmax/LayerNorm over hidden with a batch split) too;
reductions over a split dim are per-die partials over
`torch.localized.views(t)` plus a fold. The runnable spectrum — including
graph capture and an `nn.Module` — lives in
`tests/stf/test_localized_map_examples.py`.

`from torch.localized import zeros` works too. For codebases that prefer
explicit imports over patching, `stf.interop.pytorch.namespace()` returns
the identical object without touching torch. `install()` refuses to clobber a
`torch.localized` that is not ours.

The localized-allocation surface (`interop.pytorch.localized_empty`, plus
the factory family `localized_zeros/ones/full` and the placement-reusing
`*_like` variants) exposes
this as `lifetime="pinned"` (CAI + registry, freed by `release()`) versus
`lifetime="gc"` (DLPack; the tensor — typically an `nn.Parameter`, where it
is the default — owns the pages, freed when the module is unloaded). See
`tests/stf/test_device_array_dlpack.py` and
`tests/stf/interop/test_localized_weights_example.py`.

## Documentation

For complete documentation, examples, and API reference, visit:

- **Full Documentation**: [nvidia.github.io/cccl/python/stf.html](https://nvidia.github.io/cccl/python/stf.html)
- **Repository**: [github.com/NVIDIA/cccl](https://github.com/NVIDIA/cccl)
- **Examples**: [github.com/NVIDIA/cccl/tree/main/python/cuda_stf/tests/stf](https://github.com/NVIDIA/cccl/tree/main/python/cuda_stf/tests/stf)
