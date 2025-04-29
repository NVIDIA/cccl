#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../python/cuda_cooperative"
python -m pip install --upgrade build
python -m build --wheel
# Move wheel to a known location for CI artifact upload
mkdir -p ../../artifacts/wheelhouse
cp dist/*.whl ../../artifacts/wheelhouse/
