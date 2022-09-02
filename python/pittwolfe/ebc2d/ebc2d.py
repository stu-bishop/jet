"""
    Model
    -----------
     A class implementing a 2D PG model of a eastern boundary currents.
"""

__author__ = 'cwolfe'

import matplotlib.pyplot as plt
from numba import jit, float64
import numpy as np
import scipy as sp
from numpy import pi

minute = 60
hour = 60*minute
day = 24*hour
year = 365*day

class Model(object):
    '''
    A class implementing a 2D PG model of a eastern boundary currents.
    '''
    
    def __init__(self, config, initial_state=None):
               
        self.dt  = config['dt']
        self.NX  = config['NX']
        self.NZ  = config['NZ']
        self.Lx  = config['Lx']
        self.H   = config['H']
        self.kv  = config['kv']
        self.kh  = config['kh']
        self.f   = config['f']
        
        self.Nsqr  = config['Nsqr']
        self.Msqr  = config['Msqr']
        self.wstar = config['wstar']
        
        if isinstance(config['Ah'], np.ndarray) and config['Ah'].ndim > 1:
            self.Ah = config['Ah'].copy()
        else:
            self.Ah = np.reshape(np.repeat(config['Ah'], self.NX*self.NZ), (self.NZ, self.NX))
        
        if isinstance(config['Ke'], np.ndarray) and config['Ke'].ndim > 1:
            self.Ke = config['Ke'].copy()
        else:
            self.Ke = np.reshape(np.repeat(config['Ke'], (self.NX+1)*(self.NZ-1)), (self.NZ-1, self.NX+1))
            
        # optional parameters        
        self.bsurf = self.get_param(config, 'bsurf', 0)
        self.kconv = self.get_param(config, 'kconv', None)
        self.advection_scheme = self.get_param(config, 'advection_scheme', 'linear')
        self.meridional_buoyancy_advection = self.get_param(config, 'meridional_buoyancy_advection', False)
        self.momentum_stepping = self.get_param(config, 'momentum_stepping', True)
        self.implicit_momentum = self.get_param(config, 'implicit_momentum', False)
        self.implicit_vertical_viscosity = self.get_param(config, 
                                            'implicit_vertical_viscosity', True)
        self.implicit_horizontal_viscosity = self.get_param(config, 
                                            'implicit_horizontal_viscosity', True)
        
        if self.implicit_momentum and not self.momentum_stepping:
            raise AttributeError('implicit_momentum = True and momentum_stepping = False are incompatible.')
            
        # select advection schemes
        if self.advection_scheme is 'linear':
            self.calc_adv = self.calc_adv_linear
        elif self.advection_scheme is 'c2':
            self.calc_adv = self.calc_adv_c2
        else:
            raise AttributeError('Unknown advection scheme.')
            
        
        self.xF =  np.linspace(-self.Lx/2, self.Lx/2, self.NX+1)
        self.zF = -np.linspace(0, self.H, self.NZ+1)

        self.xC = .5*(self.xF[1:] + self.xF[:-1])
        self.zC = .5*(self.zF[1:] + self.zF[:-1])

        self.maskF = np.ones((self.NZ, self.NX+1))
        self.maskF[:, 0] = 0
        self.maskF[:,-1] = 0

        self.dx = self.Lx/self.NX
        self.dz = self.H/self.NZ
        
        # background buoyancy field due to Nsqr
        self.bbar = np.tile(self.Nsqr*(self.zC[:,np.newaxis] + self.H), self.NX)
        
        if initial_state is None:
            self.set_initial_condition(b=None, u=None, v=None, 
                Gn_b =None, Gn_u =None, Gn_v =None,    
                Gn1_b=None, Gn1_u=None, Gn1_v=None,
                w=None, phi=None)
        else:
            self.set_initial_condition(**initial_state)
        
    
    def get_param(self, config, key, default):
        try:
            return config[key]
        except KeyError:
            return default    
    
    def set_initial_condition(self, t=None, 
        b=None, u=None, v=None, 
        Gn_b =None, Gn_u =None, Gn_v =None,    
        Gn1_b=None, Gn1_u=None, Gn1_v=None,
        w=None, phi=None):
        
        if t is None:
            self.t = 0
        else:
            self.t = t
    
        if b is None:
            self.b = np.zeros((self.NZ, self.NX))
        else:
            self.b = b.copy()
    
        if u is None:
            self.u = np.zeros((self.NZ, self.NX+1))
        else:
            self.u = u.copy()
    
        if v is None:
            self.v = np.zeros((self.NZ, self.NX+1))
        else:
            self.v = v.copy()
            
    
        if Gn_b is None:
            self.Gn_b = np.zeros((self.NZ, self.NX))
        else:
            self.Gn_b = Gn_b.copy()
    
        if Gn1_b is None:
            self.Gn1_b = np.zeros((self.NZ, self.NX))
        else:
            self.Gn1_b = Gn1_b.copy()
    
        if not self.implicit_momentum:
            if Gn_u is None:
                self.Gn_u = np.zeros((self.NZ, self.NX+1))
            else:
                self.Gn_u = Gn_u.copy()
    
            if Gn_v is None:
                self.Gn_v = np.zeros((self.NZ, self.NX+1))
            else:
                self.Gn_v = Gn_v.copy()
            
            
            if Gn1_u is None:
                self.Gn1_u = np.zeros((self.NZ, self.NX+1))
            else:
                self.Gn1_u = Gn1_u.copy()
    
            if Gn1_v is None:
                self.Gn1_v = np.zeros((self.NZ, self.NX+1))
            else:
                self.Gn1_v = Gn1_v.copy()
            
        
        # these are not technically state variables
        if w is None:
            self.w = np.zeros((self.NZ+1, self.NX))
        else:
            self.w = w.copy()
    
        if phi is None:
            self.phi = np.zeros((self.NZ, self.NX))
        else:
            self.phi = phi.copy()
            
            
    # Dynamics
    def integrate_for_pressure(self):
        '''
        We use the "finite difference" form from the MITgcself.
        '''
    
        dzM = np.repeat(.5*self.dz, self.NZ)
        dzM[0] = self.zF[0] - self.zC[0]
    
        dzP = np.repeat(.5*self.dz, self.NZ)
        dzP[-1] = self.zC[-1] - self.zF[-1]
    
        phiC = np.zeros((self.NZ, self.NX))
        phiF = np.zeros((self.NZ+1, self.NX))

        for k in range(self.NZ):
            phiC[k,:]   = phiF[k,:] - dzM[k]*self.b[k,:]
            phiF[k+1,:] = phiC[k,:] - dzP[k]*self.b[k,:]
    
        self.phi = phiC
        
        
    def grad_phi(self):
        '''
        Calculate grad phi from phi
        '''
        phix = np.zeros((self.NZ, self.NX+1))
    
        # Neumann BC
        phix[:,1:-1] = np.diff(self.phi, axis=1)/self.dx
    
        phiy = np.zeros((self.NZ, self.NX+1))
        phiy[:,1:-1] = np.repeat(-self.Msqr*(self.zC[:,np.newaxis] + self.H/2), 
            self.NX-1, axis=1)
    
        return phix, phiy
        
        
    def calc_visc_x(self, vel):
        visc_flx_x = self.Ah*np.diff(vel, axis=1)/self.dx
    
        visc_x = np.zeros((self.NZ, self.NX+1))
        visc_x[:,1:-1] = np.diff(visc_flx_x, axis=1)/self.dx
    
        return visc_x
        
        
    def calc_visc_x_implicit(self, vel):
    
        D = 1 + self.dt*(self.Ah[:,1:] + self.Ah[:,:1])/self.dx**2
        U = -self.dt*self.Ah[:,1:-1]/self.dx**2

        visc_x = np.zeros((self.NZ, self.NX+1))
        visc_x[:,1:-1] = solve_sym_tridiagonal(D.T, U.T, vel[:,1:-1].T).T

        return visc_x
        
        
    def calc_vertical_viscosity(self):
        return self.Ke*self.f**2/self.Nsqr

        
    def calc_visc_z(self, vel):
        Av = self.calc_vertical_viscosity()
    
        visc_flx_z = np.zeros((self.NZ+1,self.NX+1))
        visc_flx_z[1:-1,:] = -Av*np.diff(vel, axis=0)/self.dz
    
        visc_z = -np.diff(visc_flx_z, axis=0)/self.dz

        return visc_z

        
    def calc_visc_z_implicit(self, vel):
        Av = self.calc_vertical_viscosity()
    
        D = np.ones([self.NZ, self.NX-1])
        D[:-1,:] += self.dt*Av[:,1:-1]/self.dz**2
        D[1:,:]  += self.dt*Av[:,1:-1]/self.dz**2
        U = -self.dt*Av[:,1:-1]/self.dz**2

        visc_z = np.zeros((self.NZ, self.NX+1))
        visc_z[:,1:-1] = solve_sym_tridiagonal(D, U, vel[:,1:-1])

        return visc_z
        
    
    def calc_uv_implicit(self, u0, v0):
        Av = self.calc_vertical_viscosity()
        
  #       build the matrices
        d = 1j*self.f + (self.Ah[:,1:] + self.Ah[:,:-1]).flatten()/self.dx**2 

        d[:self.NX-1] += Av[0,1:-1].flatten()/self.dz**2
        d[self.NX-1:-(self.NX-1)] += (Av[1:,1:-1] + Av[:-1,1:-1]).flatten()/self.dz**2
        d[-(self.NX-1):] += (Av[-1,1:-1]).flatten()/self.dz**2

        u1 = np.zeros((self.NX-1)*self.NZ-1)
        l1 = np.zeros((self.NX-1)*self.NZ-1)
        for k in range(self.NZ):
            idx = np.arange(self.NX-2) + (self.NX-1)*k
            u1[idx] = -self.Ah[k,1:-1]/self.dx**2
            l1[idx] = -self.Ah[k,1:-1]/self.dx**2
    
        u2 = -Av[:,1:-1].flatten()/self.dz**2
        l2 = -Av[:,1:-1].flatten()/self.dz**2

        A = sp.sparse.diags([l2, l1, d, u1, u2], [-self.NX+1, -1, 0, 1, self.NX-1], format='csc')
        preconditioner
        M = sp.sparse.diags([1/d], [0], format='csc')

#         N = self.NZ*(self.NX-1)
#         A = np.zeros((N, N), dtype='complex128')
# 
#         row = 0
#         for k in range(self.NZ):
#             for i in range(self.NX-1):
#                 vel = np.zeros((self.NZ, self.NX+1))
#                 vel[k,i+1] = 1
# 
#                 G = -1j*self.f*vel + self.calc_visc_x(vel) + self.calc_visc_z(vel)
#                 A[row,:] = -G[:,1:-1].flatten()
#                 row += 1
    
        self.integrate_for_pressure()
        phix, phiy = self.grad_phi()
        rhs = -(phix + 1j*phiy)
        rhs -= np.mean(rhs, axis=0, keepdims=True)
        rhs = rhs[:,1:-1].flatten()

        u = np.zeros((self.NZ, self.NX+1))
        v = np.zeros((self.NZ, self.NX+1))

        q0 = (u0 + 1j*v0)[:,1:-1].flatten()

        q, info = sp.sparse.linalg.cgs(A, rhs, q0, M=M, tol=1e-14, maxiter=1000)
#         q = sp.sparse.linalg.spsolve(A, rhs)
        
        u[:,1:-1] = np.reshape(q.real, (self.NZ, self.NX-1))
        v[:,1:-1] = np.reshape(q.imag, (self.NZ, self.NX-1))        
        
#         from IPython.core.debugger import Tracer
#         Tracer()()
        return u, v
        
        
    def step_uv_implicit(self, u0, v0):
        Av = self.calc_vertical_viscosity()
        
  #       build the matrices
        d = 1 - self.dt*(1j*self.f + (self.Ah[:,1:] + self.Ah[:,:-1]).flatten()/self.dx**2)

        d[:self.NX-1] -= self.dt*Av[0,1:-1].flatten()/self.dz**2
        d[self.NX-1:-(self.NX-1)] -= self.dt*(Av[1:,1:-1] + Av[:-1,1:-1]).flatten()/self.dz**2
        d[-(self.NX-1):] -= self.dt*(Av[-1,1:-1]).flatten()/self.dz**2

        u1 = np.zeros((self.NX-1)*self.NZ-1)
        l1 = np.zeros((self.NX-1)*self.NZ-1)
        for k in range(self.NZ):
            idx = np.arange(self.NX-2) + (self.NX-1)*k
            u1[idx] = self.dt*self.Ah[k,1:-1]/self.dx**2
            l1[idx] = self.dt*self.Ah[k,1:-1]/self.dx**2
    
        u2 = self.dt*Av[:,1:-1].flatten()/self.dz**2
        l2 = self.dt*Av[:,1:-1].flatten()/self.dz**2

        A = sp.sparse.diags([l2, l1, d, u1, u2], [-self.NX+1, -1, 0, 1, self.NX-1], format='csc')
        # preconditioner
        M = sp.sparse.diags([1/d], [0], format='csc')
    
        self.integrate_for_pressure()
        phix, phiy = self.grad_phi()
        rhs = -self.dt*(phix + 1j*phiy) + u0 + 1j*v0
        rhs -= np.mean(rhs, axis=0, keepdims=True)
        rhs = rhs[:,1:-1].flatten()

        u = np.zeros((self.NZ, self.NX+1))
        v = np.zeros((self.NZ, self.NX+1))

        q0 = (u0 + 1j*v0)[:,1:-1].flatten()

#         q, info = sp.sparse.linalg.cgs(A, rhs, q0, M=M, tol=1e-14, maxiter=1000)
        q = sp.sparse.linalg.spsolve(A, rhs)

        u[:,1:-1] = np.reshape(q.real, (self.NZ, self.NX-1))
        v[:,1:-1] = np.reshape(q.imag, (self.NZ, self.NX-1))        
        
#         from IPython.core.debugger import Tracer
#         Tracer()()
        return u, v
        
    # Continuity
    def calc_w(self):
        '''
        Calculate w from u using the continuity equation
        '''
        ux = np.diff(self.u, axis=1)/self.dx
    
        self.w = np.zeros((self.NZ+1, self.NX))
    
        self.w[-2::-1,:] = -np.cumsum(ux[::-1,:]*self.dz, axis=0)
        
    
    # Thermodynamics
    def calc_adv_linear(self):
        '''
        Calculate the time tendency of b due to advection from the velocities 
        and previous value of b
        '''
        
        flx_x = np.zeros((self.NZ, self.NX+1))
        flx_x[:,1:-1] = self.u[:,1:-1]*(self.bbar[:,1:] +self.bbar[:,:-1])/2
    
        flx_z = np.zeros((self.NZ+1, self.NX))
        flx_z[1:-1,:] = self.w[1:-1,:]*(self.bbar[1:,:] + self.bbar[:-1,:])/2
    
        div_flx_x =  np.diff(flx_x, axis=1)/self.dx
        div_flx_z = -np.diff(flx_z, axis=0)/self.dz
    
        adv = div_flx_x + div_flx_z
        
        if self.meridional_buoyancy_advection:
            adv += -self.Msqr*(self.v[:,1:] + self.v[:,:-1])/2
    
        return -adv        


    def calc_adv_c2(self):
        '''
        Calculate the time tendency of b due to advection from the velocities 
        and previous value of b
        '''
        
        flx_x = np.zeros((self.NZ, self.NX+1))
        flx_x[:,1:-1] = self.u[:,1:-1]*(self.b[:,1:] + self.b[:,:-1])/2
    
        flx_z = np.zeros((self.NZ+1, self.NX))
        flx_z[1:-1,:] = self.w[1:-1,:]*(self.b[1:,:] + self.b[:-1,:])/2
    
        div_flx_x =  np.diff(flx_x, axis=1)/self.dx
        div_flx_z = -np.diff(flx_z, axis=0)/self.dz
    
        adv = div_flx_x + div_flx_z
        
        if self.meridional_buoyancy_advection:
            adv += -self.Msqr*(self.v[:,1:] + self.v[:,:-1])/2
    
        return -adv        


    def calc_diff_x(self):
        '''
        Calculate time tendency of b due to diffusion
        '''
    
        flx_x = np.zeros((self.NZ, self.NX+1))
        flx_x[:,1:-1] = self.kh*np.diff(self.b, axis=1)/self.dx
    
        div_flx_x = np.diff(flx_x, axis=1)/self.dx
    
        return div_flx_x
     
        
    def calc_diff_z(self):
        '''
        Calculate time tendency of b due to diffusion
        '''
    
        flx_z = np.zeros((self.NZ+1, self.NX))
        flx_z[1:-1,:] = -self.kv*np.diff(self.b, axis=0)/self.dz

        div_flx_z = -np.diff(flx_z, axis=0)/self.dz
        
        return div_flx_z
     
    def calc_diff_z_implicit(self):
        '''
        Calculate time tendency of b due to diffusion
        '''
        
        bz = -np.diff(self.b, axis=0)/self.dz
        
        self.kv_tot = np.zeros_like(bz)
        self.kv_tot[bz >= 0] = self.kv
        self.kv_tot[bz <  0] = self.kconv
            
        D = np.ones([self.NZ, self.NX])
        D[:-1,:] += self.dt*self.kv_tot/self.dz**2
        D[1:,:]  += self.dt*self.kv_tot/self.dz**2
        U = -self.dt*self.kv_tot/self.dz**2

        b2 = solve_sym_tridiagonal(D, U, self.b)
    
        return b2
     
    
    def calc_surface_forcing(self):
        return self.wstar*(self.bsurf - self.b[0,:])/self.dz
        
        
    # Time stepping
    def adams_bashforth3(self, Gn, Gn1, Gn2):
        G = (23*Gn - 16*Gn1 + 5*Gn2)/12
    
        return G, Gn, Gn1
        
        
    def adams_bashforth2(self, Gn, Gn1):
        eps_ab = 0.01
        G = (1.5 + eps_ab)*Gn - (0.5 + eps_ab)*Gn1
    
        return G, Gn
            
            
    def timestep(self):
        if self.momentum_stepping:
            # Coriolis
            self.Gn_u =  self.f*self.v
            self.Gn_v = -self.f*self.u
        
            # horizontal velocity
            if not self.implicit_horizontal_viscosity:
                self.Gn_u += self.calc_visc_x(self.u)
                self.Gn_v += self.calc_visc_x(self.v)
            
            # vertical velocity
            if not self.implicit_vertical_viscosity:
                self.Gn_u += self.calc_visc_z(self.u)
                self.Gn_v += self.calc_visc_z(self.v)

            # extrapolate
            G_u, self.Gn1_u = self.adams_bashforth2(self.Gn_u, self.Gn1_u)
            G_v, self.Gn1_v = self.adams_bashforth2(self.Gn_v, self.Gn1_v)

            self.integrate_for_pressure()
            phix, phiy = self.grad_phi()

            # timestep velocity
            self.u += self.dt*self.maskF*(G_u - phix)
            self.v += self.dt*self.maskF*(G_v - phiy)
        
            if self.implicit_horizontal_viscosity:
                self.u = self.calc_visc_x_implicit(self.u)
                self.v = self.calc_visc_x_implicit(self.v)
        
            if self.implicit_vertical_viscosity:
                self.u = self.calc_visc_z_implicit(self.u)
                self.v = self.calc_visc_z_implicit(self.v)
        
            # apply surface correction
            self.u -= np.mean(self.u, axis=0, keepdims=True)
            self.v -= np.mean(self.v, axis=0, keepdims=True)

        # thermodyanmcs
        self.calc_w()

        self.Gn_b  = self.calc_diff_x()
        if self.kconv is None:
            self.Gn_b += self.calc_diff_z()
        self.Gn_b += self.calc_adv() 
        self.Gn_b[0,:] += self.calc_surface_forcing()

        G_b, self.Gn1_b = self.adams_bashforth2(self.Gn_b, self.Gn1_b)
        self.b += self.dt*G_b
        
        if self.kconv is not None:
            self.b = self.calc_diff_z_implicit()

        self.t += self.dt
        
        
    def timestep_implicit_momentum(self):
        self.u, self.v = self.step_uv_implicit(self.u, self.v)

        # thermodyanmcs
        self.calc_w()

        diff_x = self.calc_diff_x()
        diff_z = self.calc_diff_z()
        self.Gn_b = (self.calc_adv() 
            + diff_x + diff_z)

        G_b, self.Gn1_b = self.adams_bashforth2(self.Gn_b, self.Gn1_b)
        self.b += self.dt*G_b

        self.t += self.dt
        
        
    def run(self, nsteps, ndiag=None, verbose=False):

    #     Tracer()()
        nsteps = int(nsteps)

        if ndiag is not None:
            ndiag = int(ndiag)
            saves = 0
            t_save = np.zeros(nsteps//ndiag)
            KE = np.zeros(nsteps//ndiag)
            APE = np.zeros(nsteps//ndiag)

        for n in range(nsteps):
            if self.implicit_momentum:
                self.timestep_implicit_momentum()
            else:
                self.timestep()
        
            if ndiag is not None and n % ndiag == 0:
                t_save[saves] = self.t
                KE[saves]  = .5*np.sum(self.u**2 + self.v**2)*self.dx*self.dz
                APE[saves] = .5*np.sum(self.b**2/self.Nsqr*self.dx*self.dz)
            
                if verbose:
                    print('t = {:.1f} days, KE = {:.5g} GJ/kg/m, APE = {:.5g} GJ/kg/m'.format(
                        t_save[saves]/3600/24, KE[saves]/1e9, APE[saves]/1e9))
                
                saves += 1
           
        if ndiag is not None:
            diags = {'t': t_save, 'KE': KE, 'APE': APE}
            return diags

    # IO
    def return_state(self):
        if self.implicit_momentum:
            state = {'t': self.t, 
                'b': self.b.copy(), 'u': self.u.copy(), 'v': self.v.copy(), 
                'w': self.w.copy(), 'phi': self.phi.copy(), 
                'Gn_b':  self.Gn_b.copy(),  'Gn_u':  self.Gn_u.copy(),  'Gn_v':  self.Gn_v.copy(), 
                'Gn1_b': self.Gn1_b.copy(), 'Gn1_u': self.Gn1_u.copy(), 'Gn1_v': self.Gn1_v.copy(), 
                }
        else:
            state = {'t': self.t, 
                'b': self.b.copy(), 'u': self.u.copy(), 'v': self.v.copy(), 
                'w': self.w.copy(), 'phi': self.phi.copy(), 
                'Gn_b':  self.Gn_b.copy(),  'Gn1_b': self.Gn1_b.copy(), 
                }
            
        return state
    
        
    def save_state(self, fname):
        state = self.return_state()
        np.savez(fname, **state)
       
        
    def load_state(self, fname):
        data = np.load(fname)
        state = dict(data)
        data.close()
        
        self.set_initial_condition(**state)


    ######## plotting routines ###########
    def plot_state(self):

        phi  = self.phi - np.mean(self.phi, axis=0, keepdims=True)

        fig = plt.figure(figsize=(12,15))
        fig.clf()
        ax = fig.subplots(nrows=5, sharex=True)

        n = 0
        fld = 100*self.u
        vmax = np.max(np.abs(fld))
        if vmax == 0:
            vmax = 1
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xF/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('$u$')

        n = 1
        fld = 100*self.v
        vmax = np.max(np.abs(fld))
        if vmax == 0:
            vmax = 1
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xF/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('v')

        n = 2
        fld = 1e6*self.w
        vmax = np.max(np.abs(fld))
        if vmax == 0:
            vmax = 1
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zF, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('$w$')

        n = 3
        fld = phi
        vmax = np.max(np.abs(fld))
        if vmax == 0:
            vmax = 1
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('$\phi$')

        n = 4
        fld = self.b
        vmax = np.max(np.abs(fld))
        if vmax == 0:
            vmax = 1
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('b')

        ax[0].set_title('t = {:.3f} years'.format(self.t/year))
        # ax[0].set_xlim([475, 525])
        ax[0].set_xlim([-500, 500])

        fig.tight_layout()
        plt.show()


    def plot_state_with_streamfunction(self):

        phi = self.phi
        phi  -= np.mean(phi, axis=0, keepdims=True)

        psi = np.zeros((NZ+1, NX+1))
        psi[1:,:] = np.cumsum(self.u, axis=0)

        fig = plt.figure(figsize=(12,12))
        fig.clf()
        ax = fig.subplots(nrows=4, sharex=True)

        n = 0
        fld = psi
        vmax = np.max(np.abs(fld))
        if vmax == 0:
            vmax = 1
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xF/1e3, self.zF, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('$\psi$')

        n = 1
        fld = self.v
        vmax = np.max(np.abs(fld))
        if vmax == 0:
            vmax = 1
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xF/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('v')

        n = 2
        fld = phi
        vmax = np.max(np.abs(fld))
        if vmax == 0:
            vmax = 1
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('$\phi$')

        n = 3
        fld = self.b
        vmax = np.max(np.abs(fld))
        if vmax == 0:
            vmax = 1
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('b')

        ax[0].set_title('t = {:.3f} years'.format(self.t/year))
        # ax[0].set_xlim([475, 525])
        ax[0].set_xlim([-500, 500])

        fig.tight_layout()
        plt.show()


    def plot_buoyancy_budget(self):
        diff_x = self.calc_diff_x()
        diff_x[0,:] = self.calc_surface_forcing()
        
        if self.kconv is None:
            diff_z = self.calc_diff_z()
        else:
            diff_z = self.calc_diff_z_implicit() - self.b
            
        adv = self.calc_adv()
        if self.meridional_buoyancy_advection:
            adv_y = self.Msqr*(self.v[:,1:] + self.v[:,:-1])/2
        else:
            adv_y = np.zeros_like(adv)
        adv_xz = adv - adv_y
        res = (adv + diff_z + diff_x)
    
        fig = plt.figure(figsize=(12,15))
        fig.clf()
        ax = fig.subplots(nrows=5, sharex=True)

        vmax = 0
        for fld in (diff_x, diff_z, adv_xz, adv_y, res):
            vmax = max(vmax, np.max(np.abs(fld*year)))
        if vmax == 0:
            vmax = 1

        n = 0
        fld = adv_xz*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('zonal/vertical advection')

        n = 1
        fld = adv_y*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('meridional advection')

        n = 2
        fld = diff_x*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('horizontal diffusion')

        n = 3
        fld = diff_z*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('vertical diffusion')

        n = 4
        fld = res*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('residual')

        ax[0].set_title('t = {:.3f} years'.format(self.t/year))
        # ax[0].set_xlim([475, 525])
        ax[0].set_xlim([-500, 500])

        fig.tight_layout()
        plt.show()
    
        return adv_xz, adv_y, diff_x, diff_z, res


    def plot_zonal_momentum_budget(self):
        if not self.implicit_horizontal_viscosity:
            visc_x = self.calc_visc_x(self.u)
        else:
            visc_x = self.calc_visc_x_implicit(self.u) - self.u
            
        if not self.implicit_vertical_viscosity:
            visc_z = self.calc_visc_z(self.u)
        else:
            visc_z = self.calc_visc_z_implicit(self.u) - self.u
        
        cori   = self.f*self.v
    
        phix, phiy = self.grad_phi()
    
        phix -= np.mean(phix, axis=0, keepdims=True)
    
        res = cori + visc_x + visc_z - phix
    
        fig = plt.figure(figsize=(12,15))
        fig.clf()
        ax = fig.subplots(nrows=5, sharex=True)

        vmax = 0
        for fld in (visc_x, visc_z, cori, phix):
            vmax = max(vmax, np.max(np.abs(fld*year)))
        if vmax == 0:
            vmax = 1

        n = 0
        fld = visc_x*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xF/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('horizontal viscosity')

        n = 1
        fld = visc_z*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xF/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('vertical viscosity')

        n = 2
        fld = cori*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xF/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('Coriolis')

        n = 3
        fld = -phix*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xF/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('pressure gradient')

        n = 4
        fld = res*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xF/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('residual')

        ax[0].set_title('t = {:.3f} years'.format(self.t/year))
        # ax[0].set_xlim([475, 525])
        ax[0].set_xlim([-500, 500])

        fig.tight_layout()
        plt.show()

        return cori, -phix, visc_x, visc_z

    def plot_meridional_momentum_budget(self):
        if not self.implicit_horizontal_viscosity:
            visc_x = self.calc_visc_x(self.v)
        else:
            visc_x = self.calc_visc_x_implicit(self.v) - self.v
            
        if not self.implicit_vertical_viscosity:
            visc_z = self.calc_visc_z(self.v)
        else:
            visc_z = self.calc_visc_z_implicit(self.v) - self.v

        cori   = -self.f*self.u
    
        phix, phiy = self.grad_phi()
    
        phiy -= np.mean(phiy, axis=0, keepdims=True)
    
        res = cori + visc_x + visc_z - phiy
    
        fig = plt.figure(figsize=(12,15))
        fig.clf()
        ax = fig.subplots(nrows=5, sharex=True)

        vmax = 0
        for fld in (visc_x, visc_z, cori, phiy):
            vmax = max(vmax, np.max(np.abs(fld*year)))
        if vmax == 0:
            vmax = 1

        n = 0
        fld = visc_x*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xF/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('horizontal viscosity')

        n = 1
        fld = visc_z*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xF/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('vertical viscosity')

        n = 2
        fld = cori*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xF/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('Coriolis')

        n = 3
        fld = -phiy*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xF/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('pressure gradient')

        n = 4
        fld = res*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xF/1e3, self.zC, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('residual')

        ax[0].set_title('t = {:.3f} years'.format(self.t/year))
        # ax[0].set_xlim([475, 525])
        ax[0].set_xlim([-500, 500])

        fig.tight_layout()
        plt.show()

        return cori, -phiy, visc_x, visc_z

    def plot_pv_budget(self):
        if not self.implicit_horizontal_viscosity:
            visc_v_x = self.calc_visc_x(self.v)
        else:
            visc_v_x = self.calc_visc_x_implicit(self.v) - self.v
            
        visc_q_x_tmp = np.diff(visc_v_x, axis=-1)/self.dx
        visc_q_x = np.zeros((self.NZ+1, self.NX))
        visc_q_x[1:-1,:] = 0.5*(visc_q_x_tmp[1:,:] + visc_q_x_tmp[:-1,:])
        visc_q_x[:,0] = 0
        visc_q_x[:,-1] = 0
        
        if not self.implicit_vertical_viscosity:
            visc_v_z = self.calc_visc_z(self.v)
        else:
            visc_v_z = self.calc_visc_z_implicit(self.v) - self.v
            
        visc_q_z_tmp = np.diff(visc_v_z, axis=-1)/self.dx
        visc_q_z = np.zeros((self.NZ+1, self.NX))
        visc_q_z[1:-1,:] = 0.5*(visc_q_z_tmp[1:,:] + visc_q_z_tmp[:-1,:])
    
        diff_b_x = self.calc_diff_x(self.b)
        diff_q_x = np.zeros((self.NZ+1, self.NX))
        diff_q_x[1:-1,:] = -np.diff(self.f*diff_b_x/self.Nsqr, axis=0)/self.dz
    
        if self.kconv is None:
            diff_b_z = self.calc_diff_z(self.b)
        else:
            diff_b_z = self.calc_diff_z_implicit(self.b) - self.b

        diff_q_z = np.zeros((self.NZ+1, self.NX))
        diff_q_z[1:-1,:] = -np.diff(self.f*diff_b_z/self.Nsqr, axis=0)/self.dz
        
        v = (self.v[:,1:] + self.v[:,:-1])/2
        vz = np.zeros((self.NZ+1, self.NX))
        vz[1:-1,:] = np.diff(v, axis=0)/self.dz
        adv_y = self.f*vz*self.Msqr/self.Nsqr
    
        res = visc_q_x + visc_q_z + diff_q_x + diff_q_z + adv_y
    
        fig = plt.figure(figsize=(12,18))
        fig.clf()
        ax = fig.subplots(nrows=6, sharex=True)

        vmax = 0
        for fld in (visc_q_x, visc_q_z, diff_q_x, diff_q_z, adv_y):
            vmax = max(vmax, np.max(np.abs(fld*year)))
        if vmax == 0:
            vmax = 1

        n = 0
        fld = visc_q_x*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zF, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('horizontal viscosity')

        n = 1
        fld = visc_q_z*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zF, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('vertical viscosity')

        n = 2
        fld = diff_q_x*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zF, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('horizontal diffusion')

        n = 3
        fld = diff_q_z*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zF, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('vertical diffusion')

        n = 4
        fld = adv_y*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zF, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('meridional advection')

        n = 5
        fld = res*year
        levels = np.linspace(-vmax, vmax, 64)
        pc = ax[n].contourf(self.xC/1e3, self.zF, fld, levels=levels, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(pc, ax=ax[n])
        ax[n].set_ylabel('residual')

        ax[0].set_title('t = {:.3f} years'.format(self.t/year))
        # ax[0].set_xlim([475, 525])
        ax[0].set_xlim([-500, 500])

        fig.tight_layout()
        plt.show()

        return visc_q_x, visc_q_z, diff_q_x, diff_q_z, adv_y, res
    
    
## utility routines
@jit(float64[:,:](float64[:,:], float64[:,:], float64[:,:]), nopython=True)
def solve_sym_tridiagonal(d, u, b):
    '''
    Solves the symmetric system Ax = b, where d and u are the main and upper diagnonals, 
    respectively.
    
    Follows the algorithm in section 2.6 of Numerical Recipes, specialized to the symmetric
    case.
    
    d, u, and b are assumed to be 2D, with the systems defined by the rows.
    '''
    
    NZ, NX = b.shape
    x = np.zeros(b.shape)
    gamma = np.zeros(b.shape)
    
    beta = d[0,:]
    x[0,:] = b[0,:]/beta
    
    for j in range(1, NZ):
        gamma[j,:] = u[j-1,:]/beta
        beta = d[j,:] - u[j-1,:]*gamma[j,:]
        x[j,:] = (b[j,:] - u[j-1,:]*x[j-1,:])/beta
        
    for j in range(NZ-1,0,-1):
        x[j-1,:] -= gamma[j,:]*x[j,:]
        
    return x