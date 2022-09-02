import numpy as np
import scipy as sp

from scipy.special import erf
from numpy import pi as π


def alpha_from_eps(ε):
    return (np.sqrt(4-ε**2) - ε)/(2-ε**2), (np.sqrt(4-ε**2) + ε)/(2-ε**2)

def qbar(y, ε, b, d):
    α1, α2 = alpha_from_eps(ε)

    f = 1 + b*y
    if d == 0:
        return f*np.piecewise(y, [y<=0, y> 0], [α1**2, α2**2])
    else:
        αsum  = (α2**2 + α1**2)/2
        αdiff = (α2**2 - α1**2)/2

        return αsum*f + αdiff*f*erf(y/d) + αdiff*b*d*np.exp(-y**2/d**2)/np.sqrt(π)

def front_fun(y, hu, ε, b, d):
    f = 1 + b*y
    q = qbar(y, ε, b, d)

    rhs = np.zeros_like(hu)
    rhs[0,:] = -ε*f*hu[1,:]
    rhs[1,:] = (f - q*hu[0,:])/ε

    return np.vstack([-ε*f*hu[1,:], (f - q*hu[0,:])/ε])

def front_bc(hua, hub, ε):
    α1, α2 = alpha_from_eps(ε)

    return np.array([hua[0] - 1/α1**2,
                     hub[0] - 1/α2**2])

def front_initial(y, ε, b):
    # use the axact solution for an unsmoothed front
    α1, α2 = alpha_from_eps(ε)

    u = np.zeros_like(y)
    h = np.zeros_like(y)

    ξ = y*(1 + b*y/2)

    h[y<=0] = (1 - ε*α1*np.exp( α1*ξ[y<=0]))/α1**2
    h[y> 0] = (1 + ε*α2*np.exp(-α2*ξ[y> 0]))/α2**2

    u[y<=0] = np.exp( α1*ξ[y<=0])
    u[y> 0] = np.exp(-α2*ξ[y> 0])

    return np.vstack([h, u])


def solve_front(ε, b, d, L0):
    from scipy.integrate import solve_bvp

    y = np.linspace(-L0, L0, 201)

    initial = front_initial(y, ε, b)
    return solve_bvp(lambda y, hu: front_fun(y, hu, ε, b, d), lambda hua, hub: front_bc(hua, hub, ε), y, initial)
