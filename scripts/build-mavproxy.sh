#!/bin/bash
#
# This script builds a standalone, portable binary for MAVProxy 1.6.2 that is
# based on Python 2.7 and pymavlink 2.0.6
#
FN_SCRIPT=$(readlink -f "$0")
DIR_SCRIPT=$(dirname "${FN_SCRIPT}")
DIR_OUT="${DIR_SCRIPT}/src/darjeeling_cmt/data"
NAME_BINARY="mavproxy"
FN_OUT="${DIR_OUT}/${NAME_BINARY}"

# first, create a temporary Python 2.7 virtual environment for MAVProxy
DIR_TEMP=$(mktemp -d)
pushd "${DIR_TEMP}"
pipenv --python 2.7 \
&& pipenv install \
      pyinstaller==3.5 \
      pymavlink==2.0.6 \
      mavproxy==1.6.2

# next, we determine the location of the "mavproxy.py" binary in that venv
pipenv run pyinstaller --clean -y "$(pipenv run which mavproxy.py)" \
  --name "${NAME_BINARY}" \
  --distpath "${DIR_OUT}" \
  --onefile \
  --hidden-import MAVProxy \
  --hidden-import MAVProxy.modules \
  --hidden-import MAVProxy.modules.mavproxy_adsb \
  --hidden-import MAVProxy.modules.mavproxy_arm \
  --hidden-import MAVProxy.modules.mavproxy_asterix \
  --hidden-import MAVProxy.modules.mavproxy_auxopt \
  --hidden-import MAVProxy.modules.mavproxy_battery \
  --hidden-import MAVProxy.modules.mavproxy_calibration \
  --hidden-import MAVProxy.modules.mavproxy_cmdlong \
  --hidden-import MAVProxy.modules.mavproxy_rally \
  --hidden-import MAVProxy.modules.mavproxy_relay \
  --hidden-import MAVProxy.modules.mavproxy_fence \
  --hidden-import MAVProxy.modules.mavproxy_param \
  --hidden-import MAVProxy.modules.mavproxy_tuneopt \
  --hidden-import MAVProxy.modules.mavproxy_mode \
  --hidden-import MAVProxy.modules.mavproxy_misc \
  --hidden-import MAVProxy.modules.mavproxy_rc \
  --hidden-import MAVProxy.modules.mavproxy_wp \
  --hidden-import MAVProxy.modules.mavproxy_log \
  --hidden-import MAVProxy.modules.mavproxy_link \
  --hidden-import MAVProxy.modules.mavproxy_terrain \
  --hidden-import MAVProxy.modules.mavproxy_misc \
  --hidden-import MAVProxy.modules.mavproxy_signing \
  --hidden-import MAVProxy.modules.mavproxy_output \
  --hidden-import pymavlink \
  --hidden-import pyserial \
  --hidden-import lxml \
  --hidden-import future \
  --hidden-import dis3 \
  --hidden-import altgraph
#  --hidden-import matplotlib

# transform the dynamically linked binary into a statically linked binary
# NOTE staticx doesn't appear to support Python 2.7 any longer
#      we should probably create a separate, temporary pipenv for
#      staticx
#   echo "transforming binary into static binary..." \
#&& pipenv run staticx "${FN_OUT}" "${FN_OUT}" \
#&& echo "transformed binary into static binary"

# ensure the temporary directory is destroyed
popd
rm -rf "${DIR_TEMP}"
