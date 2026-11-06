#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <image> <host repository path> <container repository path>" >&2
  exit 1
fi

image="$1"
host_repo="$(cygpath -am "$2")"
mount_repo="$(cygpath -am "$3")"
host_repo_posix="$(cygpath -au "$host_repo")"
probe_dir="$(mktemp -d "${host_repo_posix}/.windows-bash-failure-probe.XXXXXX")"
trap 'rm -rf -- "$probe_dir"' EXIT
probe_name="${probe_dir##*/}"

cat > "$probe_dir/fail.sh" <<'EOF'
#!/usr/bin/env bash
cmake.exe --version > cmake-version.txt
cmake.exe -E false | tee native-failure.log
touch must-not-exist
EOF

cat > "$probe_dir/probe.ps1" <<'EOF'
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& bash --noprofile --norc -e -o pipefail ./fail.sh
exit $LASTEXITCODE
EOF

echo "Checking native failure propagation through Bash, PowerShell, and Docker."
docker_exit=0
# Docker consumes native Windows paths; do not let Git Bash rewrite the mounts.
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' docker run --rm \
  --mount "type=bind,source=${host_repo},target=${mount_repo}" \
  --workdir "$mount_repo" \
  --isolation=process \
  "$image" powershell "${mount_repo}/${probe_name}/probe.ps1" || docker_exit=$?

if [[ "$docker_exit" -ne 1 ]]; then
  echo "Error: expected Docker exit code 1 from the native failure, got $docker_exit." >&2
  exit 1
fi

# These files distinguish the intended failure from a container startup error.
if [[ ! -s "$probe_dir/cmake-version.txt" || ! -f "$probe_dir/native-failure.log" ]]; then
  echo "Error: the negative control did not reach the native CMake failure pipeline." >&2
  exit 1
fi

if [[ -e "$probe_dir/must-not-exist" ]]; then
  echo "Error: Bash continued after the failing native command pipeline." >&2
  exit 1
fi

cat "$probe_dir/cmake-version.txt"
echo "Negative control passed: Docker returned 1 and Bash stopped at the native failure."
