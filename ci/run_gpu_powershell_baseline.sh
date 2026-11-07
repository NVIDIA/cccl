#!/usr/bin/env bash
# Experiment-only native PowerShell baseline for the two cudax GPU crashes.
set -euo pipefail
exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./ci/run_cudax_powershell_baseline.ps1
