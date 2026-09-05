#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ci_dir/test_cuda_coop_python.sh" "$@"
