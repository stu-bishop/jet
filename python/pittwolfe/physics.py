"""
    physics
    -----------
    A collection of routines for physics and physical parameterizations
"""
import numpy as np
import scipy as sp
import xarray as xr

def clausius_clapeyron(T):
    '''We use the August-Roche-Magnus formula, which returns vapor pressure in mb given temperature in Celcius.'''

    return 6.1094*np.exp(17.625*T/(T + 243.04))

def dewpoint_to_specific_humidity(p,Td):
    '''Get specific humidity from pressure (in mb) and dew point (in K)'''

    Rd = 287. # gas constant for dry air
    Rv = 461. # gas constant for water vapor
    T0 = 273.15 # conversion from C to K

    e = clausius_clapeyron(Td - T0)

    r = (Rd/Rv)*(e/(p-e))
    q = r/(1+r)

    return q

def bulkformulae(theta, atemp, aqh, uwind, vwind):
    '''Large and Yeager 2004 bulk formulae taken from MITgcm EXF package'''
    nIter_bulk = 2

    cen2kel        =      273.150 # conversion from Celcius to Kelvin
    gravity_mks    =        9.81  # gravity
    atmrho         =        1.220 # density of air
    atmcp          =     1005.000 # heat capacity of air
    flamb          =  2500000.000
    flami          =   334000.000
    cvapor_fac     =   640380.000 # Coeff to calculate Saturation Specific Humidity (Gill 1982, p. 41, eq. 3.1.15)
    cvapor_exp     =     5107.400
    cvapor_fac_ice = 11637800.000
    cvapor_exp_ice =     5897.800
    humid_fac      =        0.608 # constant entering the evaluation of the virtual temperature
    gamma_blk      =        0.010 # adiabatic lapse rate
    saltsat        =        0.980 # reduction of saturation vapor pressure over salt water
    cdrag_1        =        0.0027000
    cdrag_2        =        0.0001420
    cdrag_3        =        0.0000764
    cstanton_1     =        0.0327
    cstanton_2     =        0.0180
    cDalton        =        0.0346
    zolmin         =     -100.000
    psim_fac       =        5.000 # coef used in turbulent fluxes calculation
    zref           =       10.000 # reference height
    hu             =       10.000 # height of mean wind
    ht             =        2.000 # height of mean temperature
    hq             =        2.    # height of specific humidity
    uMin           =        0.5
    karman         =        0.4 # von Karman constant
    rhoConstFresh  =      999.  # density of fresh water
    cen2kel        = 273.15
    exf_scal_BulkCdn =          1 # Drag coefficient scaling factor
    recip_rhoConstFresh = 1./rhoConstFresh

    # surface parameters
    zwln = np.log(hu/zref)
    ztln = np.log(ht/zref)
    czol = hu*karman*gravity_mks

    # compute turbulent surface fluxes
    Tsf    = theta + cen2kel
    deltap = atemp + gamma_blk*ht - Tsf

    tmpbulk = ma.array(zeros(Tsf.shape),mask=Tsf.mask)
    tmpbulk[~Tsf.mask] = cvapor_fac*exp(-cvapor_exp/Tsf[~Tsf.mask])
    ssq     = saltsat*tmpbulk/atmrho
    delq    = aqh - ssq

    # initial guess for exchange coefficients:
    # take U_N = del.U ; stability from del.Theta ;
    stable = 0.5 + np.copysign(0.5,deltap)

    # Solve for stress
    wSpeed = sqrt(uwind**2 + vwind**2)
    sh = np.maximum(wSpeed, uMin)
    wsm = sh
    tmpbulk = exf_scal_BulkCdn * ( cdrag_1/wsm + cdrag_2 + cdrag_3*wsm )
    rdn     = np.sqrt(tmpbulk)
    ustar   = rdn*wsm

    # initial guess for exchange other coefficients
    rhn = (1.-stable)*cstanton_1 + stable*cstanton_2
    ren = cDalton

    # calculate turbulent scales
    tstar = rhn*deltap
    qstar = ren*delq

    for iter in range(nIter_bulk):
        # iterate with psi-functions to find transfer coefficients
        t0 = atemp * (1. + humid_fac*aqh)
        huol = ( tstar/t0 + qstar/(1./humid_fac + aqh))*czol/ustar**2
        tmpbulk = np.minimum(np.abs(huol),10.)
        huol   = np.copysign(tmpbulk , huol)
        htol   = huol*ht/hu
        hqol   = huol*hq/hu
        stable = 0.5 + np.copysign(0.5,huol)

        # Evaluate all stability functions assuming hq = ht.
        psimh = ma.array(np.zeros(huol.shape),mask=huol.mask)
        psixh = ma.array(np.zeros(htol.shape),mask=htol.mask)

        xsq    = np.sqrt( np.abs(1. - huol*16.) )
        x      = np.sqrt(xsq)
        psimh[~huol.mask] = -psim_fac*huol[~huol.mask]*stable[~huol.mask] \
            + (1.-stable[~huol.mask])*(log((1. + 2.*x[~huol.mask] + xsq[~huol.mask]) \
            * (1.+xsq[~huol.mask])*.125) - 2.*np.arctan(x[~huol.mask]) + 0.5*pi )

        xsq    = np.sqrt( np.abs(1. - htol*16.) )
        psixh[~htol.mask] = -psim_fac*htol[~htol.mask]*stable[~htol.mask] \
            + (1.-stable[~htol.mask])*( 2.*np.log(.5*(1. + xsq[~htol.mask])) )

        # Shift wind speed using old coefficient
        dzTmp = (zwln-psimh)/karman

        old_settings = np.seterr(divide='ignore') # masked arrays through a ridiculous divide-by-zero error
        usn   = wSpeed/(1. + rdn*dzTmp )
        np.seterr(**old_settings);

        usm   = maximum(usn, uMin)

        # Update the 10m, neutral stability transfer coefficients (momentum)
        tmpbulk    = exf_scal_BulkCdn *( cdrag_1/usm + cdrag_2 + cdrag_3*usm )
        rdn        = np.sqrt(tmpbulk)
        rd         = rdn/(1. + rdn*dzTmp)
        ustar      = rd*sh
        tau        = atmrho*rd*wSpeed

        # Update the 10m, neutral stability transfer coefficients (sens&evap)
        rhn = (1.-stable)*cstanton_1 + stable*cstanton_2
        ren = cDalton

        # Shift all coefficients to the measurement height and stability.
        rh = rhn/(1. + rhn*(ztln-psixh)/karman)
        re = ren/(1. + ren*(ztln-psixh)/karman)

        # Update ustar, tstar, qstar using updated, shifted coefficients.
        qstar = re*delq
        tstar = rh*deltap


    # Turbulent Fluxes
    hs = atmcp*tau*tstar # sensible heat flux into ocean
    hl = flamb*tau*qstar # latent heat flux into ocean

    # change sign and convert from kg/m^2/s to m/s via rhoConstFresh
    evap = -recip_rhoConstFresh*tau*qstar

    tmpbulk =  tau*rd
    ustress = tmpbulk*uwind
    vstress = tmpbulk*vwind

    return hs, hl, evap, ustress, vstressdd

def baroclinic_modes(b, dzF):
    '''
    Calculate a set of baroclinic modes from a buoyancy profile.

    Parameters:
    -----------
    b : (K,) array-like
        The buoyancy on C-grid centers with depth increasing as the index
        increases.
    dzF : (K,) array-like
        Spacing between C-grid faces.

    Returns:
    --------
    c : (K-1,) ndarray
        Baroclinic wave speeds in descending order.
    S : (K+1, K-1) ndarray
        Matrix of vertical velocity modes with depth as rows and mode number as
        columns
    R : (K, K-1) ndarray
        Matrix of pressure modes with depth as rows and mode number as columns

    Notes:
    ------
    Following the notation of Ferrari et al. (2010), the vertical modes for
    pressure and vertical velocity, $R_m$ and $S_m$, respectively, satisfy

    $$\begin{aligned}
        \frac{d}{dz}\left(\frac{1}{N^2}\frac{dR_m}{dz}\right) + \frac{R_m}{c_m^2} &= 0 \\
        \frac{dR_m}{dz}(0) = \frac{dR_m}{dz}(-H) &= 0
    \end{aligned}$$

    and

    $$\begin{aligned}
        \frac{d^2S_m}{dz^2} + \frac{N^2}{c_m^2}S_m &= 0 \\
        S_m(0) = S_m &= 0
    \end{aligned}$$

    The two sets of modes are related by
    $$\begin{aligned}
        R_m &= c_m\frac{dS_m}{dz}, \\
        N^2 S_m &= -\frac{dR_m}{dz}.
    \end{aligned}$$

    The eigenvectors are orthonormal in the sense that
    $$\begin{aligned}
        \int_{-H}^0 S_m N^2 S_n \, dz &= \delta_{mn}, \\
        \int_{-H}^0 R_m R_n \, dz &= \delta_{mn}.
    \end{aligned}$$

    Given that $N^2$ may be zero in places (especially since we set $N^2 = 0$
    for $N^2 < 0$), it is easier to solve the problem for the vertical velocity
    modes and recover the pressure modes by differentiation. If we pick a
    vertical grid where $\Delta z^f$ is the distance between $w$-points and
    $\Delta z^c$ is the distance between $p$-points, we have

    $$\begin{aligned}
        N^2_k &= \frac{b_k - b_{k+1}}{\Delta z^c_k}, \\
        \frac{d^2 S}{dz^2}\Bigg|_k &= \frac{1}{\Delta z^c_k}
        \left(\frac{S_{k-1} - S_k}{\Delta z^f_{k}} - \frac{S_{k} -
            S_{k+1}}{\Delta z^f_{k+1}}\right)
    \end{aligned}$$
    We can therefore write the numerical problem as a generalized eigenvalue
    problem of the form $\mathrm{B}\boldsymbol{S} = c^2\mathrm{D}\boldsymbol{S},$
    where
    $$\begin{aligned}
        D_{kk} &= \frac{1}{\Delta z^f_k} + \frac{1}{\Delta z^f_{k+1}}, \\
        D_{k,k+1} &= D_{k+1,k} = -\frac{1}{\Delta z^f_{k+1}}, \\
        B_{kk} &= b_k - b_{k+1}.
    \end{aligned}$$

    Note that multiplying through by $\Delta z^c$ makes the system symmetric
    and guarentees orthogonal eigenvectors.
    '''

    Δb = -np.diff(b)
    Δb[Δb < 0] = 0

    K = len(dzF)

    akp1 = -1/dzF[1:-1]
    ak = 1/dzF[1:] + 1/dzF[:-1]

    # second derivative matrix
    D = np.diag(ak) + np.diag(akp1, 1) + np.diag(akp1,-1)

    # straification matrix
    B = np.diag(Δb)

    λ, vecs = sp.linalg.eigh(B, D)

    # this routine very occationally produces a negative eigenvalue when it should produce a zero eigenvalue
    λ[λ < 0] = 0

    c = np.sqrt(λ)

    # sort
    idx = np.argsort(c)[::-1]
    c = c[idx]
    vecs = vecs[:,idx]

    # eigenvectors come out orthonormal wrt to D. Renormalize to be orthonormal wrt to B.
    # be careful to about zero eigenvalues
    recip_c = np.ones_like(c)
    recip_c[c > 0] = 1/c[c > 0]

    vecs = vecs@np.diag(recip_c)

    # add back the zeros at the top and the bottom
    S = np.zeros((K+1, K-1))
    S[1:-1,:] = vecs

    # calculate the pressure modes
    R = -(np.diff(S, axis=0)/dzF[:,np.newaxis])@np.diag(c)

    return c, S, R

def getPhi(b, grid):
    '''
    Calculate hydrostatic pressure anomaly from buoyancy.

    Parameters:
    -----------
    b : DataArray with coordinates ('Z', 'YC', 'XC')
        The buoyancy field.
    grid : Dataset
        Dataset containing grid descriptors drC, drF, Z, Zp1, YC, and XC.

    Returns:
    --------
    phiC : DataArray with coordinates ('Z', 'YC', 'XC')
        Hydrostatic pressure anomaly at cell centers.
    phiF : DataArray with coordinates ('Zp1', 'YC', 'XC')
        Hydrostatic pressure anomaly at cell faces.

    Note:
    -----
    This is essentially the calculation from calc_phi_hyd in the MITgcm but it does not add
    the barotropic contribution from the free surface.
    '''

    nz, ny, nx = b.shape

    dRlocM = 0.5*grid.drC[:-1].values
    dRlocM[0] = (grid.Zp1[0]-grid.Z[0]).values
    dRlocP = 0.5*grid.drC[1:].values
    dRlocP[-1] = (grid.Z[-1] - grid.Zp1[-1]).values

    phiC = xr.DataArray(np.zeros([nz,   ny, nx]), coords=(grid.Z,   grid.YC, grid.XC), dims=('Z',   'YC', 'XC'))
    phiF = xr.DataArray(np.zeros([nz+1, ny, nx]), coords=(grid.Zp1, grid.YC, grid.XC), dims=('Zp1', 'YC', 'XC'))

    # integrate from the top with the BC phi(z=0) = 0
    for k in np.arange(0,nz):

        phiC[k,:,:]   = phiF[k,:,:] - dRlocM[k]*b[k,:,:]
        phiF[k+1,:,:] = phiC[k,:,:] - dRlocP[k]*b[k,:,:]

    return phiC, phiF
