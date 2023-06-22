#! /bin/bash

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
source ${SCRIPT_PATH}/configuration.bash

${SCRIPT_PATH}/clear_docker_containers_and_images.bash

# If invoked with an argument, the argument is the name of the tar.bz2 to import.
if [ "" == "${1}" ]
then
  TARBZ2="${BASE_NAME}.tar.bz2"
else
  TARBZ2="${1}"
fi

bunzip2 ${TARBZ2}
if [ "${?}" != "0" ]; then exit 1; fi

docker load --input ${TARBZ2%.tar.bz2}.tar
if [ "${?}" != "0" ]; then exit 1; fi

${SCRIPT_PATH}/attach_cuda_driver_to_docker_image.bash
if [ "${?}" != "0" ]; then exit 1; fi

docker image rm ${BASE_IMAGE}

rm -f ${TARBZ2%.tar.bz2}.tar

