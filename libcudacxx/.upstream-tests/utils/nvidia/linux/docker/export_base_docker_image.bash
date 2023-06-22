#! /bin/bash

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
source ${SCRIPT_PATH}/configuration.bash

${SCRIPT_PATH}/clear_docker_containers_and_images.bash

# Arguments are a list of SM architectures to target; if there are no arguments,
# all known SM architectures are targeted.

LIBCUDACXX_COMPUTE_ARCHS="${@}" ${SCRIPT_PATH}/build_base_docker_image.bash
if [ "${?}" != "0" ]; then exit 1; fi

docker save --output ${BASE_NAME}.tar ${BASE_IMAGE}
if [ "${?}" != "0" ]; then exit 1; fi

bzip2 --force ${BASE_NAME}.tar
if [ "${?}" != "0" ]; then exit 1; fi

docker image rm ${BASE_IMAGE}
docker image rm ${HOST_OS_KIND}:${HOST_OS_VERSION}

