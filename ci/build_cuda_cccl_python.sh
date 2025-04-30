#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../python/cuda_cccl"
python -m pip install --upgrade build
python -m build --wheel
# Move wheel to a known location for CI artifact upload
mkdir -p ../../wheelhouse
cp dist/*.whl ../../wheelhouse/
