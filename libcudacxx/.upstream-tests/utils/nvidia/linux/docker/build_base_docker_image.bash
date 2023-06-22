#! /bin/bash

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
source ${SCRIPT_PATH}/configuration.bash

# Arguments are a list of SM architectures to target; if there are no arguments,
# all known SM architectures are targeted.

# Pull the OS image from Docker Hub; try a few times in case our connection is
# slow.
docker pull ${OS_IMAGE} \
|| docker pull ${OS_IMAGE} \
|| docker pull ${OS_IMAGE} \
|| docker pull ${OS_IMAGE}

# Copy the .dockerignore file from //sw/gpgpu/libcudacxx to //sw/gpgpu.
cp ${LIBCUDACXX_PATH}/docker/.dockerignore ${LIBCUDACXX_PATH}/..

LIBCUDACXX_COMPUTE_ARCHS="${@}" docker -D build \
  --build-arg LIBCUDACXX_SKIP_BASE_TESTS_BUILD \
  --build-arg LIBCUDACXX_COMPUTE_ARCHS \
  --build-arg COMPILER_CXX_DIALECT=$COMPILER_CXX_DIALECT \
  -t ${BASE_IMAGE} \
  -f ${BASE_DOCKERFILE} \
  ${LIBCUDACXX_PATH}/.. 2>&1 \
  | while read l; do \
      echo "${LIBCUDACXX_DOCKER_OUTPUT_PREFIX}$(date --rfc-3339=seconds)| $l"; \
    done
if [ "${PIPESTATUS[0]}" != "0" ]; then exit 1; fi

# Create a temporary container so we can extract the log files.
TMP_CONTAINER=$(docker create ${BASE_IMAGE})

docker cp ${TMP_CONTAINER}:/sw/gpgpu/libcudacxx/build/logs.tar .

tar -xf logs.tar
rm logs.tar

docker container rm ${TMP_CONTAINER} > /dev/null

# Remove the .dockerignore from //sw/gpgpu.
rm -f ${LIBCUDACXX_PATH}/../.dockerignore

