#! /bin/bash

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
source ${SCRIPT_PATH}/configuration.bash

LIBCUDACXX_SKIP_BASE_TESTS_BUILD=1 ${SCRIPT_PATH}/build_base_docker_image.bash

${SCRIPT_PATH}/attach_cuda_driver_to_docker_image.bash
if [ "${?}" != "0" ]; then exit 1; fi

${SCRIPT_PATH}/run_docker_image.bash ${@}
if [ "${?}" != "0" ]; then exit 1; fi


