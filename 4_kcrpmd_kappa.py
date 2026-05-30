"""
Part 4 of full calculation

This script reads in dynamical trajectories from 3_kcrpmd_dynamics.py,
from the ensemble of trajectories, the transmission coefficient is computed and saved

"""

import os
import h5py
import numpy as np
from liblibra_core import *

# Number of blocks for standard error analysis
N_blocks = 10

# Check current directory name
current_dir = os.path.basename(os.getcwd())
if current_dir.startswith("_fix_"):
    fix = current_dir[current_dir.find("_fix_") + len("_fix_")]
else:
    print("not in correct directory, directory should start with '_fix_'.")
    exit()

# Sorting itraj directories by value
itraj_dirs = sorted([d for d in os.listdir('libra_data/') if d.startswith('_itraj_')])

# Initializing position and momentum data list of reaction coordinate (s or y)
pos_data_list = []
mom_data_list = []
# Loop through _itraj_* directories and collect reaction coordinate data
for i, d in enumerate(itraj_dirs):
    if fix == 's':
        with h5py.File("libra_data/" + d + "/mem_data.hdf", 'r') as f:
            time = f["time/data"][:]
            s = f["q/data"][:, 0, 0]
            ps = f["p/data"][:, 0, 0]
        pos_data_list.append(s)
        mom_data_list.append(ps)
    elif fix == 'y':
        with h5py.File("libra_data/" + d + "/mem_data.hdf", 'r') as f:
            time = f["time/data"][:]
            y = f["y_aux_var/data"][:, 0]
            py = f["p_aux_var/data"][:, 0]
        pos_data_list.append(y)
        mom_data_list.append(py)

# Stacking list of arrays to be 2D array where axis 0 are the trajectories and axis 1 are the time points
pos_data = np.vstack(pos_data_list)
mom_data = np.vstack(mom_data_list)

xi_pts =  pos_data.shape[0]
t_pts = pos_data.shape[1]

# Numerator and Denominator blocks of the transmission coefficient definition
Num_blocks = np.zeros((N_blocks, t_pts))
Den_blocks = np.zeros(N_blocks)

block_index = np.arange(xi_pts).astype(int)
block_index = block_index.reshape(N_blocks, int(xi_pts / N_blocks))

# Evaluating the kappa expression for each block. A single loop over i takes a whole block
# of point and evaluates the time dependent numerator average Num_blocks[i,:] -> (t_pts),
# then the denominator average is taken at the end which is time independent Den_blocks[i] -> ().
# the time dependent kappa coefficeint of each block i is then kappa_i(t) = Num_blocks[i,:] / Den_blocks[i]
for i in range(N_blocks):
    Den = 0.
    for k in block_index[i]:
        if mom_data[k,0] >= 0.:
            Den += mom_data[k,0]
    Den_blocks[i] = Den

    for j in range(t_pts):
        Num = 0.
        for k in block_index[i]:
            xit = pos_data[k,j]
            if xit >= 0.:
                Num += mom_data[k,0]
        Num_blocks[i,j] = Num

# Taking average over blocks, the full kappa coefficient is then kappa(t) = A / B
A = np.mean(Num_blocks, axis = 0)
B = np.mean(Den_blocks)

# Taking standard deviation of each block, this is to compute standard error of kappa
sA = np.std(Num_blocks, axis = 0)
sB = np.std(Den_blocks)

# Full time dependent kappa coefficient
kappa_avg = A / B
kappa_avg[0] = 1.

# Standard error of full time dependent kappa coefficient
kappa_se = kappa_avg * np.sqrt((sA / A)**2 + (sB / B)**2) / np.sqrt(N_blocks - 1)
kappa_se[0] = 0.

# Saving kappa transmission coefficient data to kappa_data/ directory
os.makedirs("kappa_data", exist_ok=True)

np.savetxt("kappa_data/time.txt", time)
np.savetxt("kappa_data/pos.txt", pos_data)
np.savetxt("kappa_data/mom.txt", mom_data)
np.savetxt("kappa_data/kappa_avg.txt", kappa_avg)
np.savetxt("kappa_data/kappa_se.txt", kappa_se)


# ======= PLOT THE RESULTS =======
import matplotlib.pyplot as plt

Nbars = 9
se_idx = (np.arange(1,Nbars) * np.floor(len(time) / Nbars)).astype(int)

plt.rcParams.update({
    'figure.figsize': (8.0, 4.0),
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'lines.linewidth': 2,
    'axes.linewidth': 3,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.direction': 'in',
    'xtick.top': True,
    'ytick.direction': 'in',
    'ytick.right': True,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'legend.frameon': False,
})

fig, ((ax1)) = plt.subplots(1,1)
ax1.set_xlabel("t (a.u.)")
ax1.set_xlim([0.0, time[-1]])
ax1.set_ylabel(r"$\kappa$(t)")
ax1.set_ylim([-0.09, 1.19])
ax1.axhline(y=0.0, color='k', linestyle='--')
ax1.plot(time, kappa_avg, color='k')
ax1.errorbar(time[se_idx], kappa_avg[se_idx], np.absolute(kappa_se[se_idx]), fmt='none', ecolor='k', elinewidth=1)
plt.savefig('transmission', bbox_inches='tight')

