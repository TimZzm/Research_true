# %%
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import copy
import h5py
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from dedalus.extras import plot_tools

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

import os
from os import listdir


# %%
# Parameters
Lx, Ly, Lz = 4,4,1
Nx, Ny, Nz = 128, 128, 32

Ra_D = -1.24e5
Prandtl = 0.7
N_s2 = 2

D_0 = 0
D_H = 1
M_0 = 0
M_H = -1

dealias = 3/2
stop_sim_time = 1500
timestepper = d3.RK222
max_timestep = 0.125
dtype = np.float64

# %%
# Bases
coords = d3.CartesianCoordinates('x','y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# %%
# Fields
p = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))
D = dist.Field(name='D', bases=(xbasis,ybasis,zbasis))
M = dist.Field(name='M', bases=(xbasis,ybasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
Z = dist.Field(name='Z', bases=zbasis)
tau_p = dist.Field(name='tau_p')
tau_B1 = dist.Field(name='tau_B1', bases=(xbasis,ybasis))
tau_B2 = dist.Field(name='tau_B2', bases=(xbasis,ybasis))
tau_D1 = dist.Field(name='tau_D1', bases=(xbasis,ybasis))
tau_D2 = dist.Field(name='tau_D2', bases=(xbasis,ybasis))
tau_M1 = dist.Field(name='tau_M1', bases=(xbasis,ybasis))
tau_M2 = dist.Field(name='tau_M2', bases=(xbasis,ybasis))
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis,ybasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis,ybasis))

# Substitutions
kappa = (Ra_D * Prandtl/((D_0-D_H)*Lz**3))**(-1/2)
nu = (Ra_D / (Prandtl*(D_0-D_H)*Lz**3))**(-1/2)

      
#Kuo_Bretherton Equilibrium

#Ra_M
Ra_M = Ra_D*(M_0-M_H)/(D_0-D_H)
G_D=(D_0-D_H)/Lz
G_M=(M_0-M_H)/Lz
Ra_BV=N_s2*Lz**4/(nu*kappa)
print(Ra_M)
print(Ra_BV)

x,y,z = dist.local_grids(xbasis,ybasis,zbasis)
Z['g']=z
Z.change_scales(3/2)

ex,ey,ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

B_op = (np.absolute(D - M - N_s2*Z)+ M + D - N_s2*Z)/2

LqW = (np.absolute(M-D+N_s2*Z)+ (M-D+N_s2*Z))/2 
# This is equivalent to max(0, M-D+N_S2*Z)
# If directly use max() or np.maximum() it will raise problem since M-D+N_s2*Z is of AddField class


Max = lambda A,B: (abs(A-N_s2*Z-B)+A-N_s2*Z+B)/2
eva = lambda A: A.evaluate()

dz= lambda A: d3.Differentiate(A, coords['z'])
dx= lambda A: d3.Differentiate(A, coords['x'])
dy= lambda A: d3.Differentiate(A, coords['y'])

ux=u@ex
uy=u@ey
uz=u@ez

ux2=ux*ux
uy2=uy*uy
uz2=uz*uz

grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_M = d3.grad(M) + ez*lift(tau_M1) # First-order reduction
grad_D = d3.grad(D) + ez*lift(tau_D1) # First-order reduction

# %%
# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, M, D, u, tau_p, tau_M1, tau_M2, tau_D1, tau_D2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p= 0")
problem.add_equation("dt(M) - kappa*div(grad_M) + lift(tau_M2) - G_M*uz= - u@grad(M)")
problem.add_equation("dt(D) - kappa*div(grad_D) + lift(tau_D2) - G_D*uz= - u@grad(D)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p)  + lift(tau_u2) = - u@grad(u)+ B_op*ez")
problem.add_equation("M(z=0) = M_0")
problem.add_equation("D(z=0) = D_0")
# The Following Defines Freeslip
problem.add_equation("uz(z=0)= 0")
problem.add_equation("uz(z=Lz)= 0")
problem.add_equation("dz(ux)(z=0)=0")
problem.add_equation("dz(uy)(z=0)=0")
problem.add_equation("dz(ux)(z=Lz)=0")
problem.add_equation("dz(uy)(z=Lz)=0")
problem.add_equation("M(z=Lz) = M_H")
problem.add_equation("D(z=Lz) = D_H")
problem.add_equation("integ(p) = 0") # Pressure gauge

# %%
# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time


# %%
# Initial condition
D.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
D['g'] *= z * (Lz - z) # Damp noise at walls
D['g'] += (D_H-D_0)*z # Add linear background
M.fill_random('g', seed=28, distribution='normal', scale=1e-3) # Random noise
M['g'] *= z * (Lz - z) # Damp noise at walls
M['g'] += (M_H-M_0)*z # Add linear background

# %%
# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
snapshots.add_task(M, name='moist buoyancy')
snapshots.add_task(D, name='dry buoyancy')
snapshots.add_task(u, name='velocity')

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
snapshots.add_task(M, name='moist buoyancy')
snapshots.add_task(D, name='dry buoyancy')
snapshots.add_task(u, name='velocity')

analysis = solver.evaluator.add_file_handler('analysis', sim_dt=0.25, max_writes=1000)
#analysis.add_task(d3.Integrate(nu*d3.div(grad_ux),('x','z')), name='fric x')
#analysis.add_task(d3.Integrate(nu*d3.div(grad_uy),('x','z')), name='fric y')
#analysis.add_task(d3.Integrate(nu*d3.div(grad_uz),('x','z')), name='fric z')

analysis.add_task(d3.Integrate(0.5 * (ux2 + uz2 + uy2),('x', 'y','z')), name='total kinetic energy')
analysis.add_task(d3.Average(0.5 * (ux2 + uz2 + uy2),('x', 'y','z')),name='mean kinetic energy')

analysis.add_task(d3.Integrate(0.5 * uz2,('x', 'y', 'z')),name='ke by uz')
analysis.add_task(d3.Integrate(0.5 * ux2,('x', 'y', 'z')),name='ke by ux')
analysis.add_task(d3.Integrate(0.5 * uy2,('x', 'y', 'z')),name='ke by uy')

analysis.add_task(d3.Integrate(uz,('x', 'y', 'z')),name='tot uz')
analysis.add_task(d3.Integrate(ux,('x', 'y', 'z')),name='tot ux')
analysis.add_task(d3.Integrate(uy,('x', 'y', 'z')),name='tot uy')

# analysis.add_task(M, name='moist buoyancy')
# analysis.add_task(D, name='dry buoyancy')

analysis.add_task(d3.Integrate(LqW, ('z')), name='integ LqW by Z')

#analysis.add_task(d3.Integrate(uz2,('z', 'x')),name='ke by z zx')
#analysis.add_task(d3.Integrate(ux2,('z', 'x')),name='ke by x zx')
#analysis.add_task(d3.Integrate(np.absolute(uz),('x', 'z')),name='sum by z xz')
#analysis.add_task(d3.Integrate(np.absolute(ux),('x', 'z')),name='sum by x xz')
#analysis.add_task(d3.Integrate(np.absolute(uz),('z', 'x')),name='sum by z zx')
#analysis.add_task(d3.Integrate(np.absolute(ux),('z', 'x')),name='sum by x zx')
#analysis.add_task(d3.Integrate(ux,('x')),name='mean ux') #正的负的加一块了……
#analysis.add_task(d3.Integrate(uy,('x')),name='mean uy')
#analysis.add_task(d3.Integrate(uz,('x')),name='mean uz')
#analysis.add_task(u, name='u')
#analysis.add_task(uz, name='uz')

# %%
# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# %%
# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')


# %%
# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()