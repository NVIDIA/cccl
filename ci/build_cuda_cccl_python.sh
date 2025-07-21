#!/bin/bash
set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage="Usage: $0 -py-version <python_version>"

if [[ $# -ne 2 || "$1" != "-py-version" ]]; then
  echo "$usage"
  exit 1
fi

py_version=$2
echo "Docker socket: " $(ls /var/run/docker.sock)

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  # Prepare mount points etc for getting artifacts in/out of the container.
  source "$ci_dir/util/artifacts/common.sh"
  # Note that these mounts use the runner (not the devcontainer) filesystem for
  # source directories because of docker-out-of-docker quirks.
  # The workflow-job GH actions make sure that they exist before running any
  # scripts.
  action_mounts=$(cat <<EOF
    --mount type=bind,source=${ARTIFACT_ARCHIVES},target=${ARTIFACT_ARCHIVES} \
    --mount type=bind,source=${ARTIFACT_UPLOAD_STAGE},target=${ARTIFACT_UPLOAD_STAGE}
EOF
)

  # Create the wheel artifact here and pass it in as an environment variable.
  # This avoids the need to pass the github auth token into the nested container.
  wheel_artifact_name=$("$ci_dir/util/workflow/get_wheel_artifact_name.sh")
else
  # If not running in GitHub Actions, we don't need to set up artifact mounts.
  action_mounts=""
  wheel_artifact_name=""
fi

# cuda_cccl must be built in a container that can produce manylinux wheels,
# and has the CUDA toolkit installed. We use the rapidsai/ci-wheel image for this.
# These images don't come with a new enough version of gcc installed, so that
# must be installed manually.
(
  set -x
  docker run --rm -i \
      --workdir /workspace/python/cuda_cccl \
      --mount type=bind,source=${HOST_WORKSPACE},target=/workspace/ \
      ${action_mounts} \
      --env py_version=${py_version} \
      --env GITHUB_ACTIONS=${GITHUB_ACTIONS:-} \
      --env GITHUB_RUN_ID=${GITHUB_RUN_ID:-} \
      --env JOB_ID=${JOB_ID:-} \
      --env WHEEL_ARTIFACT_NAME=${wheel_artifact_name:-} \
      rapidsai/ci-wheel:cuda12.9.0-rockylinux8-py3.10 \
      /workspace/ci/build_cuda_cccl_wheel.sh
)
