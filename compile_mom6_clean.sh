#!/bin/bash

# Quick start:
# 1. Make the directory build/{platform}/ocean_only/uv_sponge and move this script to that directory.
# 2. Make sure the MOM6_EXAMPLES variable points to the correct directory.
# 3. Pick a mkmf template and make sure the TEMPLATE variable points to it.
# 4. Run this script in the directory build/{platform}/ocean_only/uv_sponge

MOM6_EXAMPLES="../../../../../MOM6-examples"
TEMPLATE="../../macOS-gnu11-openmpi.mk"
MKMF_PATH="$MOM6_EXAMPLES/src/mkmf/bin"

rm -f path_names
$MKMF_PATH/list_paths -l ./ $MOM6_EXAMPLES/src/MOM6/{config_src/infra/FMS2,config_src/memory/dynamic_symmetric,config_src/drivers/solo_driver,config_src/external,src/{*,*/*}}
$MKMF_PATH/mkmf -t $TEMPLATE -o '-I../../shared/fms2' -p MOM6 -l '-L../../shared/fms2 -lfms' path_names
make NETCDF=4 MOM6 -j

