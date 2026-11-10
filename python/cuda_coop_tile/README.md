# cuda-coop-tile

Cooperative algorithms for cuTile Python and C++ kernels, backed by CUB.
Python programs use `import cuda.coop.tile as coop`; C++ consumers include
`<cuda/coop/tile.cuh>` and link the shared TileLib.

The Python distribution includes the frontend, the TileLib, C++ headers and
CMake package, and its own CCCL headers. The `cuda.coop` namespace uses PEP 420
so packages can contribute modules independently.

Building requires CUDA 13.5 and the separately supplied Tile/SIMT preview SDK.
The SDK is a build input and is not included in the source or wheel payload.
Configure `CUDA_COOP_TILE_SDK_ROOT` for the SDK and
`CUDAToolkit_ROOT` for the toolkit, and
`CUDA_COOP_TILE_INTERFACE_TRANSLATOR` for the `cuda-tile-translate` executable
built from the public CUDA Tile 13.5 release. See the
[Tile documentation](https://nvidia.github.io/cccl/unstable/tile/index.html)
for installation, programming guides, and supported operations.
