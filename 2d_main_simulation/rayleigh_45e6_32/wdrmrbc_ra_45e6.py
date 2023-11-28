# %%
"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.
This script demonstrates solving a 2D Cartesian initial value problem. It can
be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_snapshots.py` script can be used to
produce plots from the saved data. It should take about 5 cpu-minutes to run.

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""


# %%
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import copy
import h5py
import numpy as np
import matplotlib
import re

import matplotlib.pyplot as plt
from dedalus.extras import plot_tools

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

import os
from os import listdir

# %%
# Parameters
Lx, Lz = 40,1
Nx, Nz = 1280, 32
Ra_M = 4.5e6
D_0 = 0
D_H = 1/3
M_0 = 0
M_H = -1
N_s2=4/3
f=0.05

Prandtl = 0.7
dealias = 3/2
stop_sim_time = 2000
timestepper = d3.RK222
max_timestep = 0.125
dtype = np.float64

# %%
# Bases
coords = d3.CartesianCoordinates('x','z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# %%
# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
D = dist.Field(name='D', bases=(xbasis,zbasis))
M = dist.Field(name='M', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
uy = dist.Field(name='uy', bases=(xbasis,zbasis))
Z = dist.Field(name='Z', bases=zbasis)
tau_p = dist.Field(name='tau_p')
tau_B1 = dist.Field(name='tau_B1', bases=xbasis)
tau_B2 = dist.Field(name='tau_B2', bases=xbasis)
tau_D1 = dist.Field(name='tau_D1', bases=xbasis)
tau_D2 = dist.Field(name='tau_D2', bases=xbasis)
tau_M1 = dist.Field(name='tau_M1', bases=xbasis)
tau_M2 = dist.Field(name='tau_M2', bases=xbasis)
tau_u1 = dist.VectorField(coords,name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords,name='tau_u2', bases=xbasis)
tau_u3 = dist.Field(name='tau_u3', bases=xbasis)
tau_u4 = dist.Field(name='tau_u4', bases=xbasis)

# Substitutions    
#Kuo_Bretherton Equilibrium
kappa = (Ra_M * Prandtl/((M_0-M_H)*Lz**3))**(-1/2)
nu = (Ra_M / (Prandtl*(M_0-M_H)*Lz**3))**(-1/2)
print('kappa',kappa)
print('nu',nu)
Td=Lz**2/(nu*kappa)**(1/2)
Tc=(Lz/(M_0-M_H))**(1/2)
Tr=1/f
R_0=Tr/Tc
print('R_0',R_0)


x,z = dist.local_grids(xbasis,zbasis)
Z['g']=z
Z.change_scales(3/2)

ex,ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

B_op = (np.absolute(D - M - N_s2*Z)+ M + D - N_s2*Z)/2

Max = lambda A,B: (abs(A-N_s2*Z-B)+A-N_s2*Z+B)/2
eva = lambda A: A.evaluate()

dz= lambda A: d3.Differentiate(A, coords['z'])
dx= lambda A: d3.Differentiate(A, coords['x'])

ux=u@ex
uz=u@ez
dxux=dx(ux)
dzux=dz(ux)
dxuz=dx(uz)
dzuz=dz(uz)

ux2=ux*ux
uy2=uy*uy
uz2=uz*uz

grad_u = d3.grad(u) + ez* lift(tau_u1) # First-order reduction
grad_ux = grad_u@ex # First-order reduction
grad_uz = grad_u@ez # First-order reduction
grad_uy = d3.grad(uy) + ez*lift(tau_u3)# First-order reduction 
grad_M = d3.grad(M) + ez*lift(tau_M1) # First-order reduction
grad_D = d3.grad(D) + ez*lift(tau_D1) # First-order reduction


# %%
# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, M, D, u, uy, tau_p, tau_M1, tau_M2, tau_D1, tau_D2, tau_u1, tau_u2, tau_u3, tau_u4], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p= 0")
problem.add_equation("dt(M) - kappa*div(grad_M) + lift(tau_M2) = - u@grad(M)")
problem.add_equation("dt(D) - kappa*div(grad_D) + lift(tau_D2) = - u@grad(D)")
problem.add_equation("dt(ux) + dx(p) - nu*div(grad_ux) + lift(tau_u2)@ex = - u@grad(ux)+f*uy")
problem.add_equation("dt(uz) + dz(p) - nu*div(grad_uz) + lift(tau_u2)@ez = - u@grad(uz) + B_op")
problem.add_equation("dt(uy) -nu*div(grad_uy) + lift(tau_u4)= -f*ux - u@grad(uy)")
problem.add_equation("uy(z=0) = 0")
problem.add_equation("dz(uy)(z=Lz) = 0")
problem.add_equation("u(z=0) = 0")
problem.add_equation("uz(z=Lz) = 0")
problem.add_equation("dz(ux)(z=Lz)=0")
problem.add_equation("M(z=0) = M_0")
problem.add_equation("D(z=0) = D_0")
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
snapshots.add_tasks(solver.state,layout='g')
#snapshots.add_task(nu*d3.div(grad_ux), name='fric x')
#snapshots.add_task(nu*d3.div(grad_uy), name='fric y')
#snapshots.add_task(nu*d3.div(grad_uz), name='fric z')

analysis = solver.evaluator.add_file_handler('analysis', sim_dt=0.1, max_writes=1000)
#analysis.add_task(d3.Integrate(nu*d3.div(grad_ux),('x','z')), name='fric x')
#analysis.add_task(d3.Integrate(nu*d3.div(grad_uy),('x','z')), name='fric y')
#analysis.add_task(d3.Integrate(nu*d3.div(grad_uz),('x','z')), name='fric z')

analysis.add_task(d3.Integrate(0.5 * (ux2 + uz2 + uy2),('x','z')), name='total kinetic energy')
analysis.add_task(d3.Integrate(0.5 * (ux2 + uz2 + uy2),('x')),name='mean kinetic energy')

analysis.add_task(d3.Integrate(0.5 * uz2,('x', 'z')),name='ke by uz')
analysis.add_task(d3.Integrate(0.5 * ux2,('x', 'z')),name='ke by ux')
analysis.add_task(d3.Integrate(0.5 * uy2,('x', 'z')),name='ke by uy')

analysis.add_task(d3.Integrate(uz,('x', 'z')),name='tot uz')
analysis.add_task(d3.Integrate(ux,('x', 'z')),name='tot ux')
analysis.add_task(d3.Integrate(uy,('x', 'z')),name='tot uy')
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
CFL = d3.CFL(solver, initial_dt=0.1, cadence=10, safety=0.4, threshold=0.05,
             max_change=1.1, min_change=0, max_dt=max_timestep)
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

    
