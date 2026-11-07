#!/usr/bin/env bash
# Experiment-only native PowerShell baseline for three memory-pool test failures.
set -euo pipefail
exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./ci/run_libcudacxx_powershell_baseline.ps1
