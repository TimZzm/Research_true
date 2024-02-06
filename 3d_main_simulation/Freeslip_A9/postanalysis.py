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
Lx, Ly, Lz = 9,9,1
Nx, Ny, Nz = 288, 288, 32

Ra_D = -1.24e5
Prandtl = 0.7
N_s2 = 2

D_0 = 0
D_H = 1
M_0 = 0
M_H = -1

dealias = 3/2
stop_sim_time = 2000
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
grad_D = d3.grad(D) + ez*lift(tau_D1) # First-order reductionpyplot.close()



folder_dir = "analysis"

file_paths = [os.path.join(folder_dir, file) for file in listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, file)) and file.endswith('.h5')]
#sort by the number in the file name
file_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
print(file_paths)


if not os.path.exists('liquid_water'):    
    os.mkdir('liquid_water')
n=0
recorded = False
for file in file_paths:
    with h5py.File(file, mode='r') as file:
        if recorded == False:
            max_level_old = np.max(file['tasks']['integ LqW by Z'])
            min_level_old = np.min(file['tasks']['integ LqW by Z'])
            recorded = True
        else:
            max_level_new = np.max(file['tasks']['integ LqW by Z'])
            min_level_new = np.min(file['tasks']['integ LqW by Z'])
            if max_level_new > max_level_old:
                max_level_old = max_level_new
            if min_level_new < min_level_old:
                min_level_old = min_level_new

for file in file_paths:
    with h5py.File(file, mode='r') as file:
        liq_wat_0 = file['tasks']['integ LqW by Z'] # This gives shape (1000,128,128, 1)
        liq_wat = liq_wat_0[:,:,:,0] # delete the redundant dimension to (1000,128,128)
        st = file['scales/sim_time']
        simtime = np.array(st)
        for t in range(len(simtime)):
            liq_wat_T=np.transpose(liq_wat[t,:,:])
            if np.max(liq_wat_T) - np.min(liq_wat_T) > 0.05:
                levels = np.arange(min_level_old, max_level_old, 0.02)
                plt.contourf(liq_wat_T, levels, cmap='Spectral_r')
            else:
                plt.contourf(liq_wat_T, cmap='Spectral_r')
            plt.colorbar(label='integrated liquid water')
            plt.xlabel('x')
            plt.ylabel('y')
            n=n+1
            # Add time title
            title = "t="+str(st[t])
            plt.title(title)
            plt.savefig(f'liquid_water/liquidwater_{"%04d" % n}.png', dpi=200,bbox_inches='tight')
            matplotlib.pyplot.close()