import numpy as np
import scipy as sp
import xarray as xr
from types import SimpleNamespace

from scipy.special import erf
from numpy import pi as π


# ---------- UnsmoothedJet ------------ #
class UnsmoothedJet(object):
    def __init__(self, δ, ε, F, b, beta_front=True, use_Δqy=True):
        '''
        Parameters
        ----------
        δ       float
                asymmetry parameter
        ε       float
                Rossby number
        F       float
                Upper layer inverse Burger number, sum of surface and first interface
                inverse Burger numbers (F = F01 + F11)
        b       float
                beta parameter
        beta_front logical, optional
                If True, use the proper form for a PV front on a beta. Default: False
        use_Δqy logical, optional
                If True, remove jump in derivative of PV as well as value. Improves convergence of
                PV smoother. Default: True

        Notes
        -----
        If beta_front == False, the velocity has the form
            u = e^{y/(1+δ)}  for y < 0
              = e^{-y/(1-δ)} for y > 0

        If beta_front == True, the velocity has the form
            u = e^{η/(1+δ)}  for y < 0
              = e^{-η/(1-δ)} for y > 0
        where η = y(1 + b y/2). In this case, the transport is only accurate to first order in b.

        It turns out not to make any practical difference which one you use if b is small.

        '''
        from scipy.optimize import root_scalar


        if np.abs(δ) > 1:
            raise RuntimeError('δ must be between -1 and 1')

        self.δ = δ
        self.ε = ε
        self.F = F
        self.b = b
        self.beta_front = beta_front
        self.use_Δqy = use_Δqy

        if beta_front:
            self.transport = 2 + b*(4*δ + ε*F*(3 + δ**2)/2)
        else:
            self.transport = 2 + b*ε*F*(3 + δ**2)/2

        if δ == 1:
            warnings.warn("Profile outcrops at y = 0. Outcrops may not be properly handled.")
        elif beta_front and ε*F > 1:
            η_outcrop = (1-δ)*np.log(ε*F*(1-δ)/(ε*F - 1))
            self.y_outcrop = (np.sqrt(1 + 2*b*η_outcrop) - 1)/b
            warnings.warn("Profile outcrops at y = {:.1f}. Outcrops may not be properly handled.".format(self.y_outcrop))

        elif ε*F > 1: # need to do more work for f-plane front
            f = lambda y: 1 + ε*F*(-1 + 2*b*δ + (1-δ)*(1 + b*y + b*(1-δ))*np.exp(-y/(1-δ)))
            fprime = lambda y: -ε*F*(1 + b*y)*np.exp(-y/(1-δ))

            y_guess = (1-δ)*np.log(ε*F*(1-δ)/(ε*F - 1))

            sol = root_scalar(f, fprime=fprime, x0=y_guess)
            if sol.converged:
                self.y_outcrop = sol.root
            else:
                raise RuntimeError("Profile outcrops, but root_scalar failed to find the outcrop position.")

            warnings.warn("Profile outcrops at y = {:.1f}. Outcrops may not be properly handled.".format(self.y_outcrop))
        else:
            self.y_outcrop = None


        # values of the thickness far from the origin
        if beta_front:
            self.Hs = 1 + ε*F
            self.Hn = 1 - ε*F
        else:
            self.Hs = 1 + ε*F*(1 - 2*b*δ)
            self.Hn = 1 - ε*F*(1 - 2*b*δ)

    def _check_y_range(self, y):
        if self.beta_front and self.b > 0 and np.any(y < -1/self.b):
            raise RuntimeError("Don't evaluate for y < -1/b!")

    def u(self, y):
        δ = self.δ

        self._check_y_range(y)

        if self.beta_front:
            y = y*(1 + self.b*y/2)

        if δ < 1:
            return np.piecewise(y,
                               [y <= 0, y > 0],
                               [
                                   lambda y: np.exp( y/(1+δ)),
                                   lambda y: np.exp(-y/(1-δ))
                               ])
        else:
            return np.piecewise(y,
                               [y <= 0, y > 0],
                               [
                                   lambda y: np.exp(y/(1+δ)),
                                   lambda y: 0
                               ])

    def z0(self, y):
        δ, b = (self.δ, self.b)

        self._check_y_range(y)

        if self.beta_front:
            η = y*(1 + b*y/2)

            if δ < 1:
                return np.piecewise(η,
                                   [η <= 0, η > 0],
                                   [
                                       lambda η:  1 - (1+δ)*np.exp( η/(1+δ)),
                                       lambda η: -1 + (1-δ)*np.exp(-η/(1-δ)),
                                   ])
            else:
                return np.piecewise(η,
                                   [η <= 0, η > 0],
                                   [
                                       lambda η:  1 - 2*np.exp( η/2),
                                       lambda η: -1
                                   ])
        else:
            if δ < 1:
                return np.piecewise(y,
                                   [y <= 0, y > 0],
                                   [
                                       lambda y: +1 - 2*b*δ - (1+δ)*(1 + b*y - b*(1+δ))*np.exp( y/(1+δ)),
                                       lambda y: -1 + 2*b*δ + (1-δ)*(1 + b*y + b*(1-δ))*np.exp(-y/(1-δ))
                                   ])
            else:
                return np.piecewise(y,
                                   [y <= 0, y > 0],
                                   [
                                       lambda y: +1 - 2*b*δ - (1+δ)*(1 + b*y - b*(1+δ))*np.exp( y/(1+δ)),
                                       lambda y: -1 + 2*b*δ
                                   ])

    def h(self, y):
        ε, F = (self.ε, self.F)

        return np.maximum(1 + ε*F*self.z0(y), 0)

    def ζ_south(self, y):
        δ = self.δ

        self._check_y_range(y)

        if self.beta_front:
            b = self.b
            f = 1 + b*y
            η = y*(1 + b*y/2)

            return f*np.piecewise(η,
                               [η <= 0, η > 0],
                               [
                                   lambda η: -np.exp(η/(1+δ))/(1+δ),
                                   lambda η: np.nan
                               ])
        else:
            return np.piecewise(y,
                               [y <= 0, y > 0],
                               [
                                   lambda y: -np.exp(y/(1+δ))/(1+δ),
                                   lambda y: np.nan
                               ])

    def ζ_north(self, y):
        δ = self.δ

        self._check_y_range(y)

        if self.beta_front:
            b = self.b
            f = 1 + b*y
            η = y*(1 + b*y/2)

            if δ < 1:
                return f*np.piecewise(η,
                                   [η < 0, η >= 0],
                                   [
                                       lambda η: np.nan,
                                       lambda η: np.exp(-η/(1-δ))/(1-δ),
                                   ])
            else:
                return np.piecewise(η,
                                   [η < 0, η >= 0],
                                   [
                                       lambda η: np.nan,
                                       lambda η: 0,
                                   ])
        else:
            if δ < 1:
                return np.piecewise(y,
                                   [y < 0, y >= 0],
                                   [
                                       lambda y: np.nan,
                                       lambda y: np.exp(-y/(1-δ))/(1-δ),
                                   ])
            else:
                return np.piecewise(y,
                                   [y < 0, y >= 0],
                                   [
                                       lambda y: np.nan,
                                       lambda y: 0,
                                   ])

    def ζ(self, y):
        return np.piecewise(y,
                           [y < 0, y == 0, y > 0],
                           [
                               lambda y: self.ζ_south(y),
                               lambda y: (self.ζ_south(y) + self.ζ_north(y))/2,
                               lambda y: self.ζ_north(y)
                           ])

    def q_south(self, y):
        b, ε = (self.b, self.ε)

        return (1 + b*y + ε*self.ζ_south(y))/self.h(y)

    def q_north(self, y):
        b, ε = (self.b, self.ε)

        return (1 + b*y + ε*self.ζ_north(y))/self.h(y)

    def q(self, y):
        b, ε = (self.b, self.ε)

        return (1 + b*y + ε*self.ζ(y))/self.h(y)

    def q_jump(self):
        δ, b, ε, F = (self.δ, self.b, self.ε, self.F)

        if self.beta_front:
            return 2*ε/((1-δ**2)*(1-ε*F*δ))
        else:
            return 2*ε/((1-δ**2)*(1-ε*F*(δ - b*(1 + δ**2))))

    def qy_jump(self):
        δ, b, ε, F = (self.δ, self.b, self.ε, self.F)

        if self.use_Δqy:
            if self.beta_front:
                return 2*ε*(ε*F*(1 + δ**2) - 2*δ + b*(1 - δ**2)*(1 - ε*F*δ))/((1 - δ**2)*(1 - ε*F*δ))**2
            else:
                return 2*ε*(ε*F*(1 - 2*b*δ)*(1 + δ**2) - 2*δ)/((1 - δ**2)*(1 - ε*F*(δ - b*(1 + δ**2))))**2
        else:
            return 0

    def q_nojump(self, y):
        return np.piecewise(y,
                           [y <= 0, y > 0],
                           [
                               lambda y: self.q_south(y),
                               lambda y: self.q_north(y) - self.q_jump() - y*self.qy_jump()
                           ])


# ---------- UnsmoothedJet ------------ #
class SmoothedJet(object):
    def __init__(self, δ, ε, F, b, l, ys=None, yn=None, beta_front=True, use_Δqy=True, dirichlet_bc=False):
        '''
        A jet with smoothed PV

        Parameters
        ----------
        δ   float
            asymmetry parameter
        ε   float
            Rossby number
        F   float
            Upper layer inverse Burger number, sum of surface and first interface
                inverse Burger numbers (F = F01 + F11)
        b   float
            beta parameter
        l   float
            smoothing width
        ys  float, optional
            Southernmost point in domain. If unset here must be set in calc_profiles. Default: None
        yn  float, optional
            Northernmost point in domain. If unset here must be set in calc_profiles. Default: None
        beta_front logical, optional
                If True, use the proper form for a PV front on a beta. Default: False
        use_Δqy logical, optional
                If True, remove jump in derivative of PV as well as value. Improves convergence of
                PV smoother. Default: True
        dirichlet_bc   logical, optional
            If true, assume the thickness reaches its asymptotic values at the boundarys. If false, use
            proper far-field boundary conditions. These give the same result if the boundaries are far away,
            but the proper boundary conditions should be better behaved if the domain is small. Default: False.

        Attributes
        ----------
        unsmoothed  UnsmoothedJet instance
                    unsmoothed jet with same parameters
        Methods
        -------
        q(y)           smoothed PV as a function of y
        calc_profiles  Calculates the internal form of h and u
        u(y)           Zonal velocity at arbitrary y
        h(y)           Thickness at arbitrary y
        ζ(y)           Vorticity at arbitrary y
        z0(y)          SSH at arbitrary y
        '''

        self.unsmoothed = UnsmoothedJet(δ, ε, F, b, beta_front=beta_front, use_Δqy=use_Δqy)

        self.δ = δ
        self.ε = ε
        self.F = F
        self.b = b
        self.l = l

        # depths far to the south and north
        self.Hs = self.unsmoothed.Hs
        self.Hn = self.unsmoothed.Hn

        self.ys = ys
        self.yn = yn

        self.dirichlet_bc = dirichlet_bc

    def q_smoothed(self, y, tol=1e-4, abs_tol=1e-4, N0=16, more_output=False):
        '''
        Smoothed PV

        Parameters
        ----------
        y       float
                Meridional position
        tol     float, optional
                Relative tolerance. Default: 1e-4
        abs_tol float
                Absolute tolerance. Default: 1e-4
        N0      int, optional
                Initial order of integration. Default: 16
        more_output   bool, optional
                If True, return more output. Default: False

        Returns
        -------
        q_smoothed       float or np.array
                Smoothed PV. Float if y is a scalar. Otherwise, np.array.
        N       array of ints, optional (if more_output = True)
                Final order of integration at each y.
        rerr    array of floats, optional (if more_output = True)
                Relative error at each y.
        rerr    array of floats, optional (if more_output = True)
                Absolute error at each y.

        Notes
        -----
        Smooths PV by convolution with a Gaussian of width l. This routine uses a couple of tricks.
         - Trick 1: We subtract the PV jump at y = 0 and calculate the convolution analytically. We do this
             because the jump would ruin the convergence of the convolution.
         - Trick 2: The convolution of the jump-free PV is performed using adaptive Gauss-Hermite quadradure. The
             order of the quadrature is doubled (starting at N0) until the relative and absolute errors between
             successive interations are smaller than their respective tolerances.
         - Trick 3: We only calculate new iterations at points where the errors are still larger than their tolerances.
             This speeds things up because most points only need N = 16 or so while trickier points need N = 1024.

        '''
        from scipy.special import erf, roots_hermite

        l = self.l

        if l == 0:
            return self.unsmoothed.q(y)
        else:
            y_is_scalar = False
            try:
                y[0]
            except TypeError:
                y_is_scalar = True

            y = np.atleast_1d(y)
            f = lambda yp: self.unsmoothed.q_nojump(yp)/np.sqrt(π)

            N = N0
            yi, wi = roots_hermite(N)

            F0 = np.sum(wi[np.newaxis,:]*f(y[:,np.newaxis] - l*yi[np.newaxis,:]), axis=1)

            rerr = np.ones_like(y)
            aerr = np.ones_like(y)

            idx = (rerr > tol) | (aerr > abs_tol)
            Ns = np.zeros_like(y)
            F1 = np.zeros_like(F0)

            while np.any(idx):
                N *= 2
                Ns[idx] = N

                yi, wi = roots_hermite(N)
                F1[idx] = np.sum(wi[np.newaxis,:]*f(y[idx,np.newaxis] - l*yi[np.newaxis,:]), axis=1)

                recip_F1 = np.abs(F1)
                recip_F1[recip_F1 == 0] = 1
                recip_F1 = 1/recip_F1

                rerr = np.abs(F1 - F0)*recip_F1
                aerr = np.abs(F1 - F0)

                idx = (rerr > tol) | (aerr > abs_tol)

                F0 = F1.copy()


            Δq = self.unsmoothed.q_jump()
            q_smooth = 0.5*Δq*(1 + erf(y/l)) + F0

            if self.unsmoothed.use_Δqy:
                Δqy = self.unsmoothed.qy_jump()
                q_smooth = q_smooth + 0.5*Δqy*(l*np.exp(-y**2/l**2)/np.sqrt(π) + y*(1 + erf(y/l)))


            if y_is_scalar:
                q_smooth = float(q_smooth)

            if more_output:
                return q_smooth, Ns, rerr, aerr
            else:
                return q_smooth

    def calc_profiles(self, ys=None, yn=None, N0=512, tol=1e-3):
        '''
        Calculates meridional profiles of smoothed jet by solving a BVP. Iterates until tol is reached.

        Parameters
        ----------
        ys  float, optional
            Southernmost point in domain. Must be set here if unset at initialization. Default: None
        yn  float, optional
            Northernmost point in domain. Must be set here if unset at initialization. Default: None
        N0  int, optional
            Initial number of grid points. Default=512
        tol float, optional
            Maximum relative difference between successive refinements
        '''

        δ, ε, F, b = self.δ, self.ε, self.F, self.b

        if ys is not None:
            self.ys = ys

        if yn is not None:
            self.yn = yn

        if self.ys is None or self.yn is None:
            raise RuntimeError('Need to set domain limits (ys, yn) before solving')


        N = N0
        err = 1

        _, _, h_last = self._solve(N)

        while (np.max(err) > tol):
            N *= 2
            yh, yq, h = self._solve(N)
            err = np.abs(h[::2] - h_last)/h[::2]
            h_last = h.copy()

        Δy = yh[1]-yh[0]
        fq = 1 + b*yq
        fh = 1 + b*yh

        # Here we're just assuming the height field assumes its far field value at the boundaries.
        u = np.zeros(N+1)
        u[1:-1] = -np.diff(h)/(fq[1:-1]*ε*F)/Δy
        if self.dirichlet_bc:
            u[0] = (self.Hs - h[0])/(ε*F*fq[0]*Δy)
            u[-1] = (h[-1] - self.Hn)/(ε*F*fq[-1]*Δy)
        else:
            u[ 0] = fh[ 0]*Δy*(self.Hs - h[ 0])/(np.sqrt(F*self.Hs)*ε*fq[ 0])
            u[-1] = fh[-1]*Δy*(h[-1] - self.Hn)/(np.sqrt(F*self.Hn)*ε*fq[-1])

        ζ = -np.diff(u)/Δy

        # find the maximum
        j = np.argmax(u)

        # refine with reverse parabolic interpolation
        du  = u[j+1] - u[j-1]
        ddu = u[j+1] - 2*u[j] + u[j-1]
        self.y0 = yq[j] - Δy*du/ddu/2
        self.umax = u[j] - du**2/ddu/8

        self.transport = np.sum(Δy*h*(u[1:] + u[:-1])/2)

        self.grid = SimpleNamespace(
            yh=yh,
            yq=yq,
            h=h,
            u=u,
            ζ=ζ,
            fq=fq,
        )

        return N, err

    def _solve(self, N):
        from scipy.linalg import solveh_banded

        ys, yn = self.ys, self.yn
        δ, ε, F, b = self.δ, self.ε, self.F, self.b

        yq = np.linspace(ys, yn, N+1)
        yh = (yq[:-1] + yq[1:])/2
        Δy = yh[1] - yh[0]

        fq = 1 + b*yq
        fh = 1 + b*yh

        Hs = self.unsmoothed.Hs
        Hn = self.unsmoothed.Hn

        qj = self.q_smoothed(yh)

        # use the "upper" diagnonal format:
        # *   a01 a12 a23 a34 a45
        # a00 a11 a22 a33 a44 a55
        A = np.zeros((2, N))

        # diagonal
        A[1,:] = 1/fq[:-1] + 1/fq[1:] + F*Δy**2*qj

        # upper diagonal
        A[0,1:] = -1/fq[1:-1]

        # rhs
        rhs = F*Δy**2*fh

        if self.dirichlet_bc:
            # If we use Dirichlet boundary conditions, only the rhs is affected:
            rhs[ 0] = rhs[ 0] + self.Hs/fq[ 0]
            rhs[-1] = rhs[-1] + self.Hn/fq[-1]
        else:
            # For Robin boundary conditions, we alter the main diagonal and the rhs
            A[1, 0] = 1/fq[ 1] + F*fh[ 0]*Δy*(Δy + np.sqrt(Hs/F)/fq[ 0])/Hs
            A[1,-1] = 1/fq[-2] + F*fh[-1]*Δy*(Δy + np.sqrt(Hn/F)/fq[-1])/Hn

            rhs[ 0] = rhs[ 0] + fh[ 0]*Δy*np.sqrt(F*Hs)/fq[ 0]
            rhs[-1] = rhs[-1] + fh[-1]*Δy*np.sqrt(F*Hn)/fq[-1]


        h = solveh_banded(A, rhs)

        return yh, yq, h

    def u(self, y, centered=True, rescaled=True):
        '''
        Zonal velocity as a function of y.

        Parameters
        ----------
        y   array-like
            Meridional positions
        centered   logical, optional
            Whether to center the jet so it has maximum velocity at y = 0. Default: True
        rescaled   logical, optional
            Whether to rescale jet so maximum velocity is 1. Default: True
        '''
        try:
            if centered:
                y0 = self.y0
            else:
                y0 = 0

            if rescaled:
                scale = 1/self.umax
                L  = 1/scale
            else:
                scale = 1
                L = 1
        except AttributeError:
            raise RuntimeError('Must call calc_profiles before evaluating profiles')

        # extrapolation with anything other than constants turns out to be overkill
        # if self.dirichlet_bc: # extrapolate using zeros
        return scale*np.interp(y/L + y0, self.grid.yq, self.grid.u, left=0, right=0)
#         else:
#             F, b, Hs = self.F, self.b, self.Hs
#             ys = self.grid.yq[ 0]
#             yn = self.grid.yq[-1]

#             ret = np.zeros_like(y)

#             # indices for interpolation
#             idx_i = (yp >= ys) & (yp <= yn)
#             ret[idx_i] = scale*np.interp(yp[idx_i], self.grid.yq, self.grid.u)

#             # south
#             idx_s = yp < ys
#             η  = yp*(1 + b*yp[idx_s]/2)
#             ηs = ys*(1 + b*ys/2)

#             ret[idx_s] = scale*self.grid.u[0]*np.exp(np.sqrt(F/Hs)*(η-ηs))

#             # north
#             idx_n = yp > yn
#             η  = yp*(1 + b*yp[idx_n]/2)
#             ηn = yn*(1 + b*yn/2)

#             ret[idx_n] = scale*self.grid.u[-1]*np.exp(np.sqrt(F/Hs)*(η-ηn))


    def ζ(self, y, centered=True, rescaled=True):
        try:
            if centered:
                y0 = self.y0
            else:
                y0 = 0

            if rescaled:
                scale = 1/self.umax
                L  = 1/scale
            else:
                scale = 1
                L = 1

            return scale**2*np.interp(y/L + y0, self.grid.yh, self.grid.ζ, left=0, right=0)

        except AttributeError:
            raise RuntimeError('Must call calc_profiles before evaluating profiles')

    def h(self, y, centered=True, rescaled=True):
        try:
            if centered:
                y0 = self.y0
            else:
                y0 = 0

            if rescaled:
                scale = 1/self.umax
                L  = 1/scale
            else:
                scale = 1
                L = 1

            return np.interp(y/L + y0, self.grid.yh, self.grid.h, left=self.Hs, right=self.Hn)

        except AttributeError:
            raise RuntimeError('Must call calc_profiles before evaluating profiles')

    def q(self, y, centered=True, rescaled=True):
        '''
        Nondimensional profile of PV

        Parameters
        ----------
        y : float
            Nondimensional meridional position
        '''

        try:
            if centered:
                y0 = self.y0
            else:
                y0 = 0

            if rescaled:
                scale = 1/self.umax
                L  = 1/scale
            else:
                scale = 1
                L = 1

            return (1 + self.b*y + self.ε*self.ζ(y, centered, rescaled))/self.h(y, centered, rescaled)

        except AttributeError:
            raise RuntimeError('Must call calc_profiles before evaluating profiles')


    def z0(self, y):
        return (self.h(y) - 1)/(self.ε*self.F)


# ---------- JetForcingAndIC ------------ #
class JetForcingAndIC(object):
    '''
    Class to set up forcing and inital conditions for jet experiments

    Attributes
    ----------
    L : float
        Length scale
    θ0 : float
        Central latitude in degrees
    f0 : float
        Coriolis parameter
    β : float
        Beta parameter
    b : float
        Beta number: b = β L/f0
    K : int
        Number of layers
    H : (K,) array of floats
        Mean thickness of the K layers
    gprime : (K,) array of floats
        Reduced gravities. gprime[0] = g
    F : (K, K) array of floats
        Inverse Burger numbers. F[i,j] = f0^2 L^2/(gprime[i]H[j])
    Umax : float
        Target maximum velocity of get in m/s
    ε : float
        Target Rossby number of jet
    l_nondim : float
        Nondimensional smoothing length
    l_dim : float
        Dimensional smoothing length in km
    recenter : bool
        Recenter jet if True
    rescale: bool
        Rescale jet if True
    Lx, Ly : float
        Zonal and meridional size of domain in km.
    transport : float
        Actual transport in Sv (should be within 1% of input transport)
    '''
    # immutable constants
    g = 9.8
    Ω = 2*π*(1 + 1/365.256363004)/24/3600 # rotation rate
    radius = 6380 # Earth radius in km
    km2m = 1e3
    small_number = 1e-20

    def __init__(self,
                 central_lat,
                 length_scale,
                 delta,
                 thicknesses,
                 smoothing_scale = 0,
                 gprime = None,
                 Fjj = None,
                 rossby_number = None,
                 umax = None,
                 transport = None,
                 Lx=None,
                 Ly=None,
                 dx=None,
                 dt=None,
                 smoothing_scale_is_nondim = True,
                 use_beta=True,
                 recenter = True,
                 rescale = True,
                 use_beta_front = True,
                 rho0 = 1035,
                ):
        '''
        Parameters
        ----------
        central_lat : float
            Central latitude in degrees
        length_scale : float
            Jet length scale in km
        thickness : (K,) array-like of floats
            List of layer thickness in m
        delta : float
            Asymmetry parameter. Should be between -1 and 1.
        smoothing_scale : float, optional
            Smoothing scale for PV. Is dimensional if smoothing_scale_is_nondim is True (the default),
            otherwise, in km. (Default is 0)
        gprime : (K-1,) array-like of floats, optional
            Reduced gravities at interfaces in m s^-2. Must specify either gprime or Fjj. (Default is None)
        Fjj : (K-1,) array-like of floats, optional
            Inverse Burger numbers of internal interfaces. Must specify either gprime or Fjj. (Default is None)
        rossby_number : float, optional
            Rossby number of flow. Must specify one of rossby_numer, umax, and transport. (Default is None)
        umax : float, optional
            Maximum zonal velocity. Must specify one of rossby_numer, umax, and transport. (Default is None)
        transport : float, optional
            Approximate total transport in Sv, T = 2ULH + O(b). Neglect O(b) term. Must specify one of rossby_numer,
            umax, and transport. (Default is None)
        Lx : float, optional
            Zonal extent of domain in km. Needed for grid. (Default is None)
        Ly : float, optional
            Meridional extent of domain in km. Needed to initialize smoothed jet profile (Default is None)
        dx : float, optional
            Grid spacing in km. Needed for grid. (Default is None)
        dt : float, optional
            Timestep in seconds. (Default is None)
        smoothing_scale_is_nondim : logical, optional
            Whether the smoothing scale is nondimensional (Default is True)
        use_beta : logical, optional
            Are we working on a beta-plane? (Default is True)
        recenter : logical, optional
            Smoothing shifts the jet axis to the south. If recenter is True, jet axis is shifted back to y = 0
            (Default is True)
        rescale : logical, optional
            Smoothing broadens the jet. Rescale so `length_scale` is still the appropriate length scale. (Default
            is True)
        use_beta_front : logical, optional
            Use beta-plane front model instead of f-plane front model. (Default is True)
        rho0 : float, optional
            Boussinesq reference density. (Default is 1035)
        '''
        self.L = length_scale
        self.δ = delta
        self.rho0 = rho0

        self.profiles_initialized = False
        self.grid_initialized = False

        self._init_rotation(central_lat, use_beta)
        self._init_stratification(thicknesses, gprime, Fjj)
        self._init_vel(rossby_number, umax, transport)

        if smoothing_scale_is_nondim:
            self.l_nondim = smoothing_scale
            self.l_dim = smoothing_scale*self.L
        else:
            self.l_nondim = smoothing_scale/self.L
            self.l_dim = smoothing_scale

        self.recenter = recenter
        self.rescale = rescale

        self.Ly = Ly
        self.Lx = Lx
        self.dx = dx
        self.dt = dt

        if Ly is not None:
            ys = -Ly/self.L
            yn =  Ly/self.L
        else:
            ys = None
            yn = None

        self.jet = SmoothedJet(self.δ, self.ε, self.F[0,0] + self.F[1,0], self.b, self.l_nondim,
                          ys=ys, yn=yn, beta_front=use_beta_front, use_Δqy=True, dirichlet_bc=False)

        if ys is not None:
            self.calc_profiles(ys, yn)

        if Lx is not None and Ly is not None and dx is not None:
            self.init_grid(Lx, Ly, dx)

        self.sponge_visc = SimpleNamespace(initialized=False)
        self.sponge_vel_rlx = SimpleNamespace(initialized=False)
        self.sponge_eta_rlx = SimpleNamespace(initialized=False)


    def _init_rotation(self, central_lat, use_beta):

        self.θ0 = central_lat
        self.f0 = 2*self.Ω*np.sin(np.radians(self.θ0))

        if use_beta:
            self.β = 2*self.Ω*np.cos(np.radians(self.θ0))/(self.radius*self.km2m)

        self.b = self.β*self.km2m*self.L/self.f0

    def _init_stratification(self, thicknesses, gprime, Fjj):
        km2m = 1e3

        self.H = np.array(thicknesses)
        self.K = len(self.H)

        if (gprime is None) ^ (Fjj is None): # XOR: one, but not both are none
            if gprime is not None:
                _gprime = np.array(gprime)

            if Fjj is not None:
                _gprime = self.f0**2*km2m**2*self.L**2/(np.array(Fjj*self.H[:-1]))
        else:
            raise RuntimeError('Must specify either gprime or Fjj')

        if len(_gprime) != self.K-1:
            raise RuntimeError('Should have one fewer value of gprime/Fjj than layers')

        self.gprime = np.insert(_gprime, 0, self.g) # first entry of gprime is g
        # half the harmonic mean of first two reduced gravities
        self.gbar = self.gprime[0]*self.gprime[1]/(self.gprime[0] + self.gprime[1])

        self.F = self.f0**2*km2m**2*self.L**2/(self.gprime[:,np.newaxis]*self.H)

    def _init_vel(self, rossby_number, umax, transport):
        '''
        Figure out Rossby number and umax
        '''

        km2m = 1e3

        if ((rossby_number is not None) + (umax is not None) + (transport is not None)) != 1: # only one is not none
            raise RuntimeError('Must specify only one of rossby_number, umax, and transport')

        if rossby_number is not None:
            self.ε = rossby_number
            self.Umax = self.f0*km2m*self.L*rossby_number

        if umax is not None:
            self.Umax = umax
            self.ε = umax/(self.f0*km2m*self.L)

        if transport is not None: # The transport is approximately 2 U L H1
            self.Umax = 1e6*transport/(2*km2m*self.L*self.H[0])
            self.ε = self.Umax/(self.f0*km2m*self.L)

    def suggest_timestep(self, dt_min, max_cfl=0.25):
        '''
        Generates a list of timesteps that evenly divide a day, are larger than dt_min,
        and produce a CFL less than a given value.

        Parameters
        ----------
        dt_min : int
            Minimum timestep (in seconds) to consider
        max_cfl : float, optional
            Maximum CFL number. (Default is 0.25)
        '''
        if self.dx is None:
            raise RuntimeError('Need to define dx first.')

        p2, p3, p5 = np.meshgrid(np.arange(8), np.arange(4), np.arange(3))
        deltas = np.unique((2**p2 * 3**p3 * 5**p5).flatten())

        cfl = self.Umax*deltas/(self.dx*self.km2m)
        return deltas[(deltas >= dt_min) & (cfl <= max_cfl)]

    def calc_profiles(self, ys=None, yn=None, N0=512, tol=1e-3):
        '''
        Calculate cross-stream profiles of velocity and thickness

        Parameters
        ----------
        ys : float, optional
            Non-dimensional location of southern boundary. If none, use value from self.Ly. (Default is None.)
        yn : float, optional
            Non-dimensional location of northern boundary. If none, use value from self.Ly. (Default is None.)
        N0 : int, optional
            Size of initial computational grid. (Default is 512.)
        tol : float, optional
            Maximum relative difference between successive approximations on finer grids. (Default is 1e-3.)
        '''
        try:
            if ys is None:
                ys = -self.Ly/self.L

            if yn is None:
                yn = self.Ly/self.L
        except TypeError:
            raise RuntimeError('Need values for ys and/or yn')

        ret = self.jet.calc_profiles(ys, yn, N0, tol)

        self.transport = self.jet.transport*self.Umax*self.H[0]*self.L/1e3

        self.profiles_initialized = True
        return ret

    def init_grid(self, Lx=None, Ly=None, dx=None):
        if Lx is None:
            Lx = self.Lx

        if Ly is None:
            Ly = self.Ly

        if dx is None:
            dx = self.dx

        # Horizontal grid
        y0 = 0

        xw = 0
        xe = xw + Lx
        ys = -Ly/2
        yn =  Ly/2

        xq = np.arange(xw, xe+dx, dx, dtype=float)
        xh = np.arange(xw+dx/2, xe, dx, dtype=float)

        yq = np.arange(ys, yn+dx, dx, dtype=float)
        yh = np.arange(ys+dx/2, yn, dx, dtype=float)

        self.NX = len(xh)
        self.NY = len(yh)

        # vertical grid
        zl = np.cumsum(self.gprime*self.rho0/self.g)
        zi = np.hstack(([1.5*zl[0] - 0.5*zl[1]],
                        (zl[:-1] + zl[1:])/2,
                        [1.5*zl[-1] - 0.5*zl[-2]]
                  ))

        self.grid = xr.Dataset({}, coords={
            'xq': (['xq'], xq),
            'xh': (['xh'], xh),
            'yq': (['yq'], yq),
            'yh': (['yh'], yh),
            'zl': (['zl'], zl),
            'zi': (['zi'], zi),
        })

        self.grid_initialized = True

        return self.grid

    # Nondimensional profiles
    def u_nondim(self, y, k=0):
        '''
        Nondimensional profile of zonal velocity

        Parameters
        ----------
        y : float
            Nondimensional meridional position
        k : integer, optional
            Layer index. (Default is 0).
        '''

        if not self.profiles_initialized:
            raise RuntimeError('Must call calc_profiles first')

        if k == 0:
            return self.jet.u(y, centered=self.recenter, rescaled=self.rescale)
        else:
            return np.zeros_like(y)

    def ζ_nondim(self, y, k=0):
        '''
        Nondimensional profile of vorticity

        Parameters
        ----------
        y : float
            Nondimensional meridional position
        k : integer, optional
            Layer index. (Default is 0).
        '''

        if not self.profiles_initialized:
            raise RuntimeError('Must call calc_profiles first')

        if k == 0:
            return self.jet.ζ(y, centered=self.recenter, rescaled=self.rescale)
        else:
            return np.zeros_like(y)

    def v_nondim(self, y, k=0):
        '''
        Nondimensional profile of meridional velocity

        Parameters
        ----------
        y : float
            Nondimensional meridional position
        k : integer, optional
            Layer index. (Default is 0).
        '''

        if not self.profiles_initialized:
            raise RuntimeError('Must call calc_profiles first')

        return np.zeros_like(y)

    def h_nondim(self, y, k=0):
        '''
        Nondimensional profile of layer thickness

        Parameters
        ----------
        y : float
            Nondimensional meridional position
        k : integer, optional
            Layer index. (Default is 0).
        '''

        if not self.profiles_initialized:
            raise RuntimeError('Must call calc_profiles first')
        # h1 = 1 + ε (F00 + F01) z0'
        # h2 = 1 - ε F11 z0' = 1 - F11 (h1 - 1)/(F00 + F01)

        if k == 0:
            return self.jet.h(y, centered=self.recenter, rescaled=self.rescale)
        elif k == 1:
            return (1 - self.F[1,1]*(self.jet.h(y, centered=self.recenter, rescaled=self.rescale) - 1)
                    / (self.F[0,0] + self.F[1,0]))
        else:
            return np.ones_like(y)

    def q_nondim(self, y, k=0):
        '''
        Nondimensional profile of PV

        Parameters
        ----------
        y : float
            Nondimensional meridional position
        k : integer, optional
            Layer index. (Default is 0).
        '''

        if not self.profiles_initialized:
            raise RuntimeError('Must call calc_profiles first')

        return (1 + self.b*y + self.ε*self.ζ_nondim(y, k))/self.h_nondim(y, k)

    def zp_nondim(self, y, k=0):
        '''
        Nondimensional profile of interface height anomaly

        Parameters
        ----------
        y : float
            Nondimensional meridional position
        k : integer, optional
            Layer index. (Default is 0).
        '''

        if not self.profiles_initialized:
            raise RuntimeError('Must call calc_profiles first')

        F = self.F[0,0] + self.F[1,0]
        ε = self.ε

        h1 = self.h_nondim(y, k=0)
        z0 = (h1 - 1)/(ε*F)

        if k == 0:
            return z0
        elif k == 1:
            return -z0
        else:
            return np.zeros_like(y)

    # Dimensional profiles
    def u(self, y, k=0):
        '''
        Dimensional profile of zonal velocity

        Parameters
        ----------
        y : float
            Meridional position in km
        k : integer, optional
            Layer index. (Default is 0).
        '''

        return self.Umax*self.u_nondim(y/self.L, k)

    def ζ(self, y, k=0):
        '''
        Dimensional profile of vorticity

        Parameters
        ----------
        y : float
            Meridional position in km
        k : integer, optional
            Layer index. (Default is 0).
        '''

        return self.Umax*self.ζ_nondim(y/self.L, k)/(self.L*self.km2m)

    def v(self, y, k=0):
        '''
        Dimensional profile of meridional velocity

        Parameters
        ----------
        y : float
            Meridional position in km
        k : integer, optional
            Layer index. (Default is 0).
        '''

        return self.Umax*self.v_nondim(y/self.L, k)

    def h(self, y, k=0):
        '''
        Dimensional profile of layer thickness

        Parameters
        ----------
        y : float
            Meridional position in km
        k : integer, optional
            Layer index. (Default is 0).
        '''

        return self.H[k]*self.h_nondim(y/self.L, k)

    def q(self, y, k=0):
        '''
        Dimensional profile of PV

        Parameters
        ----------
        y : float
            Meridional position in km
        k : integer, optional
            Layer index. (Default is 0).
        '''

        return self.f0*self.q_nondim(y/self.L, k)/self.H[k]

    def zp(self, y, k=0):
        '''
        Dimensional profile of interface height anomaly

        Parameters
        ----------
        y : float
            Meridional position in km
        k : integer, optional
            Layer index. (Default is 0).
        '''

        if k < 2:
            return self.f0*self.Umax*self.L*self.km2m/self.gprime[k]*self.zp_nondim(y/self.L, k)
        else:
            return np.zeros_like(y)

    def z(self, y, k=0):
        '''
        Dimensional profile of interface height anomaly

        Parameters
        ----------
        y : float
            Meridional position in km
        k : integer, optional
            Layer index. (Default is 0).
        '''

        return -self.H[:k].sum() + self.zp(y, k)

    def viscosity_from_cfl(self, viscous_cfl):
        '''
        Sets the viscosity so that the viscous CFL is the given value.
        '''

        Δx = self.dx*self.km2m # grid spacing in meters
        Kh = viscous_cfl*Δx**2/(4*self.dt)

        self.sponge_visc.Kh = Kh
        return Kh

    def set_viscous_sponge(self,
                                 widthE, rampE,
                                 widthW, rampW,
                                 widthN, rampN,
                                 widthS, rampS,
                                 Kh=None,
                                 field_name='Kh'
                          ):
        '''
        Set parameters of viscous sponge

        Parameters
        ----------
        widthN/S/E/W : float
            Width (in km) of part of sponge with constant viscosity.
        rampN/S/E/W : float
            Width (in km) of part of sponge with linearly ramping viscosity.
        Kh : float, optional
            Sponge viscosity. Can be set by viscosity_from_cfl. (Default is None--does nothing)
        field_name : string, optional
            Name of viscous sponge field in input netCDF. (Default is 'Kh')
        '''

        self.sponge_visc.initialized = True
        self.sponge_visc.widthN = widthN
        self.sponge_visc.rampN  = rampN
        self.sponge_visc.widthS = widthS
        self.sponge_visc.rampS  = rampS
        self.sponge_visc.widthE = widthE
        self.sponge_visc.rampE  = rampE
        self.sponge_visc.widthW = widthW
        self.sponge_visc.rampW  = rampW

        if Kh is not None:
            self.sponge_visc.Kh = Kh

        self.sponge_visc.field_name = field_name

        return self.sponge_visc

    def set_relax_sponge_vel(self,
                                widthE, rampE,
                                widthW, rampW,
                                widthN=0, rampN=0,
                                widthS=0, rampS=0,
                                rlx_time=1,
                                field_name_u='Idamp_u',
                                field_name_v='Idamp_v'
                          ):
        '''
        Set parameters of velocity relaxation sponge

        Parameters
        ----------
        widthE/W : float
            Width (in km) of part of sponge with constant relaxation rate.
        rampE/W : float
            Width (in km) of part of sponge with linearly ramping relaxation rate.
        widthN/S : float, optional
            Width (in km) of part of sponge with constant relaxation rate. (Default is 0)
        rampN/S : float
            Width (in km) of part of sponge with linearly ramping relaxation rate. (Default is 0)
        rlx_time : float, optional
            Relaxation time scale in days. (Default is 1)
        field_name_u/v : string, optional
            Name of damping rate field in input netCDF. (Default is 'Idamp_u/v')
        '''

        self.sponge_vel_rlx.initialized = True
        self.sponge_vel_rlx.widthN = widthN
        self.sponge_vel_rlx.rampN  = rampN
        self.sponge_vel_rlx.widthS = widthS
        self.sponge_vel_rlx.rampS  = rampS
        self.sponge_vel_rlx.widthE = widthE
        self.sponge_vel_rlx.rampE  = rampE
        self.sponge_vel_rlx.widthW = widthW
        self.sponge_vel_rlx.rampW  = rampW

        self.sponge_vel_rlx.max_rate = 1/(rlx_time*3600*24) # Convert inverse days to inverse seconds

        self.sponge_vel_rlx.field_name_u = field_name_u
        self.sponge_vel_rlx.field_name_v = field_name_v

        return self.sponge_vel_rlx

    def set_relax_sponge_eta(self,
                                widthE, rampE,
                                widthW, rampW,
                                widthN=0, rampN=0,
                                widthS=0, rampS=0,
                                rlx_time=30,
                                field_name_damp='Idamp_eta',
                                field_name_target='eta_target'
                          ):
        '''
        Set parameters of interface relaxation sponge

        Parameters
        ----------
        widthE/W : float
            Width (in km) of part of sponge with constant relaxation rate.
        rampE/W : float
            Width (in km) of part of sponge with linearly ramping relaxation rate.
        widthN/S : float, optional
            Width (in km) of part of sponge with constant relaxation rate. (Default is 0)
        rampN/S : float
            Width (in km) of part of sponge with linearly ramping relaxation rate. (Default is 0)
        rlx_time : float, optional
            Relaxation time scale in days. (Default is 1)
        field_name_damp : string, optional
            Name of damping rate field in input netCDF. (Default is 'Idamp_eta')
        field_name_target : string, optional
            Name of target height field in input netCDF. (Default is 'eta_target')
        '''

        self.sponge_eta_rlx.initialized = True
        self.sponge_eta_rlx.widthN = widthN
        self.sponge_eta_rlx.rampN  = rampN
        self.sponge_eta_rlx.widthS = widthS
        self.sponge_eta_rlx.rampS  = rampS
        self.sponge_eta_rlx.widthE = widthE
        self.sponge_eta_rlx.rampE  = rampE
        self.sponge_eta_rlx.widthW = widthW
        self.sponge_eta_rlx.rampW  = rampW

        self.sponge_eta_rlx.max_rate = 1/(rlx_time*3600*24) # Convert inverse days to inverse seconds

        self.sponge_eta_rlx.field_name_damp = field_name_damp
        self.sponge_eta_rlx.field_name_target = field_name_target

        return self.sponge_eta_rlx

    def _sponge_EW(self, x, sponge):
        xW = self.grid.xq[ 0]
        xE = self.grid.xq[-1]

        damp = 0*x

        damp[x < xW + sponge.widthW] = 1

        if sponge.rampW > 0:
            x1 = xW + sponge.widthW
            x2 = xW + sponge.widthW + sponge.rampW
            idx = (x1 <= x) & (x < x2)
            damp[idx] = 1-(x[idx]-x1)/(x2-x1)

        damp[x >= xE - sponge.widthE] = 1

        if sponge.rampE > 0:
            x1 = xE - sponge.widthE - sponge.rampE
            x2 = xE - sponge.widthE
            idx = (x1 <= x) & (x < x2)
            damp[idx] = (x[idx]-x1)/(x2-x1)

        return damp

    def _sponge_NS(self, y, sponge):
        yS = self.grid.yq[ 0]
        yN = self.grid.yq[-1]

        damp = 0*y

        damp[y < yS + sponge.widthS] = 1

        if sponge.rampS > 0:
            y1 = yS + sponge.widthS
            y2 = yS + sponge.widthS + sponge.rampS
            idx = (y1 <= y) & (y < y2)
            damp[idx] = 1-(y[idx]-y1)/(y2-y1)

        damp[y >= yN - sponge.widthN] = 1

        if sponge.rampN > 0:
            y1 = yN - sponge.widthN - sponge.rampN
            y2 = yN - sponge.widthN
            idx = (y1 <= y) & (y < y2)
            damp[idx] = (y[idx]-y1)/(y2-y1)

        return damp

    def viscous_sponge(self):
        if not self.sponge_visc.initialized:
            raise RuntimeError('Sponge not initialized. Call set_viscous_sponge first.')

        return self.sponge_visc.Kh*np.maximum(
                            self._sponge_NS(self.grid.yh, self.sponge_visc),
                            self._sponge_EW(self.grid.xh, self.sponge_visc)
                        )

    def relax_sponge_vel(self):
        if not self.sponge_vel_rlx.initialized:
            raise RuntimeError('Sponge not initialized. Call set_relax_sponge_vel first.')

        idamp_u = self.sponge_vel_rlx.max_rate*np.maximum(
                            self._sponge_NS(self.grid.yh, self.sponge_vel_rlx),
                            self._sponge_EW(self.grid.xq, self.sponge_vel_rlx)
                        )

        idamp_v = self.sponge_vel_rlx.max_rate*np.maximum(
                            self._sponge_NS(self.grid.yq, self.sponge_vel_rlx),
                            self._sponge_EW(self.grid.xh, self.sponge_vel_rlx)
                        )

        return idamp_u, idamp_v

    def relax_sponge_eta(self):
        if not self.sponge_eta_rlx.initialized:
            raise RuntimeError('Sponge not initialized. Call set_relax_sponge_eta first.')

        return self.sponge_eta_rlx.max_rate*np.maximum(
                            self._sponge_NS(self.grid.yh, self.sponge_eta_rlx),
                            self._sponge_EW(self.grid.xh, self.sponge_eta_rlx)
                        )

    # put initial conditions in an xarray
    def to_xarray(self):

        ret = self.grid.copy()

        ret['u'] = xr.concat([
                        xr.DataArray(
                            data=self.u(self.grid.yh, k=k),
                            dims=['yh'],
                            coords=dict(
                                yh=self.grid.yh
                            )
                        )
                        for k in range(self.K)
                    ], 'zl') + 0*ret.xq

        ret['v'] = xr.concat([
                        xr.DataArray(
                            data=self.v(self.grid.yq, k=k),
                            dims=['yq'],
                            coords=dict(
                                yq=self.grid.yq
                            )
                        )
                        for k in range(self.K)
                    ], 'zl') + 0*ret.xh

        ret['eta'] = xr.concat([
                        xr.DataArray(
                            data=self.z(self.grid.yh, k=k),
                            dims=['yh'],
                            coords=dict(
                                yh=self.grid.yh
                            )
                        )
                        for k in range(self.K+1)
                    ], 'zi') + 0*ret.xh

        if self.sponge_visc.initialized:
            ret[self.sponge_visc.field_name] = self.viscous_sponge()

        if self.sponge_vel_rlx.initialized:
            idamp_u, idamp_v = self.relax_sponge_vel()
            ret[self.sponge_vel_rlx.field_name_u] = idamp_u
            ret[self.sponge_vel_rlx.field_name_v] = idamp_v

        if self.sponge_eta_rlx.initialized:
            ret[self.sponge_eta_rlx.field_name_damp] = self.relax_sponge_eta()
            # The way mom does sponges is weird. Let η(k) be the instantaneous height and ηs(k)
            # be the reference height. MOM6 compares (η(0) - η(k))/(η(0) + H) to ηs(k)/ηs(K). This
            # means we need to subtract the surface height from all the reference values.

            # Not sure if we should make sure the lowest level target should be the actual bottom or
            # the bottom minus the surface.

            # We need to scale up the interior heights and set the free surface height to zero.
            ret[self.sponge_eta_rlx.field_name_target] = ret.eta - ret.eta.isel(zi=0)

        return ret

