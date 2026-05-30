"""
Part 3 of full calculation

This script is takes as argument:
    --itraj: index of trajectory
    --istart: starting index of configuration to be pulled from thermalization data
    --iskip: index spacing between configuration to be pulled from thermalization data separating trajectories

This script reads in the _control_params_dynamics.txt generated from 1_kcrpmd_tst.py,
then it reads in a configuration from thermalization calculation (part 2).
From the sampled configurations and the control parameters, dynamical trajectories are generated

"""

import os
import h5py
import numpy as np
import argparse

from liblibra_core import *
from libra_py import units
import libra_py.dynamics.tsh.compute as tsh_dynamics

from kcrpmd_utils.KCRPMD_System_Bath import kcrpmd_system_bath

parser = argparse.ArgumentParser()
parser.add_argument('--itraj', default=0, type=int, help='transmission trajectory index')
parser.add_argument('--istart', default=999, type=int, help='thermalization starting index')
parser.add_argument('--iskip', default=49, type=int, help='thermalization skipping index')
parser.add_argument('--nsteps', default=125000, type=int, help='total number of timesteps')
parser.add_argument('--nprint', default=500, type=int, help='number of timesteps between prints')
args = parser.parse_args()

pref = F"_itraj_{args.itraj}"

# Check current directory name
current_dir = os.path.basename(os.getcwd())
if current_dir.startswith("libra_data"):
    pass
else:
    print("not in correct directory, directory should be libra_data")
    exit()

###################################################################################
# ======= READ IN KC-RPMD RECIPE, MODEL, AND INITIAL CONDITIONS: DYNAMICS ======= #
###################################################################################

with open("../_control_params_dynamics.txt") as f:
    control_params = eval(f.read())

with open("../_model_params.txt") as f:
    model_params = eval(f.read())

model_params.update({"hw": 0})
control_params.update({ "prefix":pref, "prefix2":pref })
control_params.update({ "nsteps":args.nsteps, "nprint":args.nprint })

########################################################################################
# ======= INITIALIZE TRANSMISSION TRAJECTORIES FROM THERMALIZATION CALCULATION ======= #
########################################################################################

with h5py.File("mem_data.hdf", 'r') as f:
    t = f["time/data"][:]
    q = list(f["q/data"][args.itraj*args.iskip + args.istart, 0, :])
    if "use_kcrpmd" in control_params:
        y = f["y_aux_var/data"][args.itraj*args.iskip + args.istart, 0]

beta = units.hartree / (units.boltzmann * control_params["Temperature"])
ms = model_params["ms"]
Mj = model_params["Mj"]
wj = model_params["wj"]
cj = model_params["cj"]
mq = model_params["mq"]

nstates = control_params["nstates"]
ntraj = control_params["ntraj"]
ndof = 2 + len(Mj)

mass = [ms] + Mj + [mq]
q = [q[0]] + [np.random.normal(loc=cj[j]*q[0]/(Mj[j]*wj[j]**2), scale=np.sqrt(1/(beta*Mj[j]*wj[j]**2))) for j in range(len(Mj))] + [q[-1]]
p = [np.random.normal(scale = np.sqrt(mass[i] / beta)) for i in range(len(mass))]

init_nucl = {"q":q, "p":p, "mass":mass, "force_constant":[0.0] * ndof, "init_type":0,
             "ntraj":ntraj, "ndof": ndof}

# Here we set the dynamical mass of y coordinate to 10 if the true TST mass is below 10.
# In this case, th final transmission coefficeint value is unchanged.
if "use_kcrpmd" in control_params:
    my = control_params["kcrpmd_my"]
    if my < 10.:
        my = 10.
    control_params["kcrpmd_gamma"] = np.sqrt(control_params["kcrpmd_my"]/my)*control_params["kcrpmd_gamma"]
    py = np.random.normal(scale=np.sqrt(my/beta))

    init_elec = {"init_type":0, "nstates":nstates, "istate":0,
                  "rep":1, "ntraj":ntraj,
                  "ndia":nstates, "nadi":nstates,
                  "y_aux_var":[y], "p_aux_var":[py], "m_aux_var":[my]}
else:
    init_elec = {"init_type":0, "nstates":nstates, "istate":0,
                  "rep":1, "ntraj":ntraj,
                  "ndia":nstates, "nadi":nstates}

rnd = Random()

res = tsh_dynamics.generic_recipe(control_params, kcrpmd_system_bath, model_params, init_elec, init_nucl, rnd)


# ======= PLOT THE RESULTS =======
import h5py
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, cumulative_trapezoid

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

# Get current directory name (last component of the path)
current_dir = os.path.basename(os.getcwd())
parent_dir = os.path.basename(os.path.dirname(os.getcwd()))
grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(os.getcwd())))

fix = parent_dir[parent_dir.find("_fix_") + len("_fix_")]

if grandparent_dir == "adiabatic":
    fig, ((ax1, ax2)) = plt.subplots(2,1,sharex=True)
    plt.subplots_adjust(hspace=0.1)
    with h5py.File(pref + "/mem_data.hdf", 'r') as f:
        t = f["time/data"][:]
        q = f["q/data"][:,0,:]

    s_low = model_params["s0"]
    s_high = model_params["s1"]
    ax1.set_ylabel(r"s (a.u.)")
    ax1.set_ylim([s_low-0.75*(s_high-s_low), s_high+0.75*(s_high-s_low)])
    ax1.axhline(y=s_low, color='k', linestyle='--')
    ax1.axhline(y=s_high, color='k', linestyle='--')
    ax1.plot(t, q[:,0], color='k')

    ax2.set_xlabel(r"t (a.u.)")
    ax2.set_ylabel(r"q (a.u.)")
    ax2.plot(t, q[:,-1], color='k')

if grandparent_dir == "kcrpmd_ori" or grandparent_dir == "kcrpmd_new":
    plt.rcParams.update({'figure.figsize': (8.0, 6.0)})
    fig, ((ax1, ax2, ax3)) = plt.subplots(3,1,sharex=True)
    plt.subplots_adjust(hspace=0.1)
    with h5py.File(pref + "/mem_data.hdf", 'r') as f:
        t = f["time/data"][:]
        y = f["y_aux_var/data"][:,0]
        q = f["q/data"][:,0,:]

    ax1.set_ylabel(r"y (a.u.)")
    ax2.set_ylim([-1.9, 1.9])
    ax1.axhline(y=-1.5, color='k', linestyle='--')
    ax1.axhline(y=-0.5, color='k', linestyle='--')
    ax1.axhline(y=0.5, color='k', linestyle='--')
    ax1.axhline(y=1.5, color='k', linestyle='--')
    ax1.plot(t, y, color='k')

    s_low = model_params["s0"]
    s_high = model_params["s1"]
    ax2.set_ylabel(r"s (a.u.)")
    ax2.set_ylim([s_low-0.75*(s_high-s_low), s_high+0.75*(s_high-s_low)])
    ax2.axhline(y=s_low, color='k', linestyle='--')
    ax2.axhline(y=s_high, color='k', linestyle='--')
    ax2.plot(t, q[:,0], color='k')

    ax3.set_xlabel(r"t (a.u.)")
    ax3.set_ylabel(r"q (a.u.)")
    ax3.plot(t, q[:,-1], color='k')

plt.savefig(pref + '/dynamics', bbox_inches='tight')

