#!/bin/bash

# Quick start:
# 1. Make the directory build/{platform}/shared/fms2 and move this script to that directory.
# 2. Make sure the MOM6_EXAMPLES variable points to the correct directory.
# 3. Pick a mkmf template and make sure the TEMPLATE variable points to it.
# 4. Run this script in the directory build/{platform}/shared/fms2

MOM6_EXAMPLES="../../../../../MOM6-examples"
TEMPLATE="../../macOS-gnu11-openmpi.mk"

MKMF_PATH="$MOM6_EXAMPLES/src/mkmf/bin"

rm -f path_names

$MKMF_PATH/list_paths $MOM6_EXAMPLES/src/FMS2

$MKMF_PATH/mkmf -c "-Duse_netCDF -Duse_libMPI -DSPMD" -t $TEMPLATE -p libfms.a path_names
make NETCDF=4 libfms.a -j

