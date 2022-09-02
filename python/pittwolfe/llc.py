"""
    llc
    -----------
    Routines for dealing with LLC grids.
"""
import numpy as np
import numpy.ma as ma


def llc_to_array(fld, cap=True):
    ntiles, nx, ny = fld.shape

    if cap:
        flat = ma.zeros((4*nx, 4*nx))
        flat[3*nx:4*nx,nx:] = ma.masked
    else:
        flat = ma.zeros((3*nx, 4*nx))

    for n in range(3):
        flat[n*nx:(n+1)*nx,0*nx:nx] = fld.isel(tile=n).values
        flat[n*nx:(n+1)*nx,1*nx:2*nx] = fld.isel(tile=n+3).values
        flat[n*nx:(n+1)*nx,2*nx:3*nx] = np.rot90(fld.isel(tile=9-n).values)
        flat[n*nx:(n+1)*nx,3*nx:4*nx] = np.rot90(fld.isel(tile=12-n).values)

    if cap:
        flat[3*nx:4*nx,:nx] = np.rot90(fld.isel(tile=6).values, 3)
        flat[3*nx:4*nx,2*nx:3*nx] = np.rot90(fld.isel(tile=6).values)

    return flat
