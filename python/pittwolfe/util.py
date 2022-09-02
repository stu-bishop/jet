"""
    util
    -----------
    A collection of random routines
"""

import numpy as np
from numpy import pi
import scipy as sp
import collections

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


### this one doesn't really work
def xrange(start, stop, step=1, include_endpoint=True):
    '''
    Clone of np.arange that works for non-integer steps and optionally includes endpoint.

    Parameters
    ----------
    start : number
        Start of interval. The interval includes this value.
    stop: number
        End of interval. Interval includes this number if include_endpoint is True and
        step evenly divides interval.
    step: number, optional
        Spacing between values. Default: 1.
    include_endpoint: bool, optional
        Whether to include endpoint. Default: True

    Returns
    -------
    xrange : ndarray
        Array of evenly spaced values.

    Notes
    -----
    Uses np.linspace under the hood.
    '''

    N = (stop-start)/step

##########################################################################################
# Reciprical trig functions calculated in a way to avoid overflow
##########################################################################################
def csc(x):
    return 1/np.sin(x)

def sec(x):
    return 1/np.cos(x)

def cot(x):
    return np.tan(np.pi/2 - x)

def csch(x):
    y = np.zeros_like(x)

    idx = np.abs(x) <= 1
    y[idx] = 1/np.sinh(idx)

    idx = x >  1
    y[idx] = 2*np.exp(-x[idx])/(1 - np.exp(-2*x[idx]))

    idx = x < -1
    y[idx] = -2*np.exp(x[idx])/(1 - np.exp(2*x[idx]))

    return y

def sech(x):
    y = np.zeros_like(x)

    idx = x >= 0
    y[idx] = 2*np.exp(-x[idx])/(1 + np.exp(-2*x[idx]))

    idx = x < 0
    y[idx] = 2*np.exp(x[idx])/(1 + np.exp(2*x[idx]))

    return y

def coth(x):
    return 1/np.tanh(x)

def arccsc(x):
    return np.arcsin(1/x)

def arcsec(x):
    return np.arccos(1/x)

def arccot(x):
    return np.arctan(1/x)


##########################################################################################
# Trig functions taking degrees as arguments
##########################################################################################
def cosd(x):
    return np.cos(np.deg2rad(x))

def sind(x):
    return np.sin(np.deg2rad(x))

def tand(x):
    return np.tan(np.deg2rad(x))

def secd(x):
    return sec(np.deg2rad(x))

def cscd(x):
    return csc(np.deg2rad(x))

def cotd(x):
    return cot(np.deg2rad(x))

def arccosd(x):
    return np.rad2deg(np.arccos(x))

def arcsind(x):
    return np.rad2deg(np.arcsin(x))

def arctand(x):
    return np.rad2deg(np.arctan(x))

##########################################################################################
# Number theory
##########################################################################################
# A function to print all prime factors of
# a given number n
def primeFactors(n):

    factors = []
    # Print the number of two's that divide n
    while n % 2 == 0:
        factors.append(2)
        n = n // 2

    # n must be odd at this point
    # so a skip of 2 ( i = i + 2) can be used
    for i in range(3,int(np.sqrt(n))+1,2):

        # while i divides n , print i ad divide n
        while n % i== 0:
            factors.append(i)
            n = n // i

    # Condition if n is a prime
    # number greater than 2
    if n > 2:
        factors.append(n)

    return factors

##########################################################################################
# numpy helper functions
##########################################################################################
def atleast_nd(x, n):
    '''
    View input as array with at least n dimensions

    Parameters
    ----------
      x   : array-like
          Array-like sequence. Non-array inputs are converted to arrays. Arrays that
          already have n dimensions are returned unchanged.
      n  : positive integer
          Desired dimension of output.

    Returns
    -------
      y : ndarray
          An array with `a.ndim == n`. New dimensions are added to the end.
    '''
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if x.ndim >= n:
        return x
    else:
        shape = list(x.shape) + [1 for n in range(n-x.ndim)]
        return x.reshape(shape)

##########################################################################################
# Linear algebra functions
##########################################################################################
def solve_symmetric_tridiagonal(a, b, rhs):
    '''
    Solve several N x N symmetric tridiagonal problems simultaneously. Works along first axis
    of inputs.

    Parameters
    ----------
      a   : ndarray
          Array of lower diagonals with `a.shape = (N, ...)`.
      b   : ndarray
          Array of diagonals with `b.shape = (N, ...)`.
      rhs : ndarray
          Array of right-hand-sides with `rhs.shape = (N, ...)`.

    Returns
    -------
      x : ndarray
          Array if solutions with `x.shape = (N, ...)`.
    '''

    sol = np.zeros_like(rhs)
    γ   = np.zeros_like(rhs)

    N = rhs.shape[0]

    # Solve many tridiagonals at once along the first axis.
    β = b[0]
    sol[0,...] = rhs[0]/β

    # decomposition and forward substitution
    for j in range(1,N):
        γ[j,...] = a[j-1]/β
        β = b[j] - a[j-1]*γ[j]
        sol[j,...] = (rhs[j] - a[j-1]*sol[j-1])/β

    # backsustitution
    for j in range(N-2,-1,-1):
        sol[j,...] -= γ[j+1]*sol[j+1]

    return sol


