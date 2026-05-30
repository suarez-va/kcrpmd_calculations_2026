"""
Part 1 of full calculation

This script is takes as argument:
    --sys: the system type (A1, A2, A3, B, or C, see KC-RPMD paper 2025)
    --meth: whether to run adi = adiabatic, ori = original KC-RPMD, new = new KC-RPMD
    --fix: which reaction coordinate to fix and evaluate rate constants from (either y or s)
    --a: KC-RPMD gaussian restraint parameter (no larger than 0.1, large enough to converge free energy of kinked-pair formation)
    --logK: Diabatic coupling constant (sys A) or prefactor (sys B, C), thermally weighted logorithmic log(beta*K_0)
    --leps: Donor-acceptor coordinate driving force (sys C only), thermally weighted beta*epsilon
    --hw: whether to include left side hard wall (-1), right side hard wall (1), or no hard wall (0)

    From kcrpmd_utils/KCRPMD_TST.py code, free energies and KC-RPMD parameters for "eta",
mass of auxiliary variable "my", and Langevin frictional coefficient "gammay",
are computed numerically and saved to _sys_*/tst_data/ for later.

    Finally, Libra control parameters dictionaries are generated and saved to _sys_*/
thermalization calculation (part 2) and for transmission coefficient dynamic trajectories (part 3) 

"""

import os
import numpy as np
import argparse

from liblibra_core import *
from libra_py import units

from kcrpmd_utils.Spin_Boson_Conventional import gen_bath_params_conventional
from kcrpmd_utils.KCRPMD_System_Bath import get_ABC, kcrpmd_system_bath
from kcrpmd_utils.KCRPMD_TST import KCRPMD_TST

######################################################
# ======= ARGUMENT PARSER, SEE TOP OF SCRIPT ======= #
######################################################
parser = argparse.ArgumentParser()
parser.add_argument('--sys', default='A1', type=str, help='System type A1, A2, A3, B, or C')
parser.add_argument('--meth', default='adi', type=str, help='Method: Adiabatic (adi), Original KC-RPMD (ori), New KC-RPMD (new)')
parser.add_argument('--fix', default='s', type=str, help='Reaction coordinate to fix: y or s')
parser.add_argument('--a', default=0.1, type=float, help='KC-RPMD gaussian restraint parameter a')
parser.add_argument('--logK', default=0.5, type=float, help='Diabatic coupling parameter, thermally weighted logorithmic log(beta*K_0)')
parser.add_argument('--leps', default=15.0, type=float, help='System type C epsilon parameter, thermally weighted beta*epsilon')
parser.add_argument('--hw', default=0, type=int, help='System type C hard wall, left side (-1), right side (1), no hard wall (0)')
args = parser.parse_args()

if (args.sys != 'A1' and args.sys != 'A2' and args.sys != 'A3' and args.sys != 'B' and args.sys != 'C'): print("Invalid System Type!"); exit() 
if (args.fix != 's' and args.fix != 'y'): print("Invalid Reaction Coordinate!"); exit() 
if (args.fix == 'y' and args.meth == 1): print("No y Coordinate For Adiabatic Method!"); exit() 
if (args.meth != 'adi' and args.meth != 'ori' and args.meth != 'new'): print("Invalid Method!"); exit() 

###########################################################################
# ======= CREATING WORKING DIRECTORIES FOR CALCULATIONS TO BE RUN ======= #
###########################################################################
if (args.sys == 'A1' or args.sys == 'A2' or args.sys == 'A3' or args.sys == 'B'):
    if (args.meth == 'ori'):
        pref = F"_fix_{args.fix}_a_{args.a}_logK_{args.logK:.2f}"
    else:
        pref = F"_fix_{args.fix}_logK_{args.logK:.2f}"
else:
    if (args.meth == 'adi'):
        pref = F"_fix_{args.fix}_logK_{args.logK:.2f}_leps_{args.leps:.2f}_hw_{args.hw}"
    elif (args.meth == 'ori'):
        pref = F"_fix_{args.fix}_a_{args.a}_leps_{args.leps:.2f}_hw_{args.hw}"
    else:
        pref = F"_fix_{args.fix}_leps_{args.leps:.2f}_hw_{args.hw}"

os.makedirs(pref, exist_ok=True)

###################################################
# ======= ASSIGNING PARAMETERS FOR SYSTEM ======= #
###################################################
T = 300.0 # Temperature in K
beta = units.hartree/(units.boltzmann*T)
a = args.a # KC-RPMD parameter a
b = 1000.0 # KC-RPMD parameter b
c = 1.0 # KC-RPMD parameter c

ms = 1836.0 # Mass of s
ws = 2.28e-3 if (args.sys=='A1' or args.sys=='B' or args.sys=='C') else 4.75e-4 # Frequency of s
s0 = -2.4 if (args.sys=='A1' or args.sys=='B' or args.sys=='C') else -8.3 # Diabat 0 parabola minima
s1 = 2.4 if (args.sys=='A1' or args.sys=='B' or args.sys=='C') else 8.3 # Diabat 1 parabola minima
eps = 0.0 # Diabat 0 to diabat 1 driving force

M = ms # Ohmic bath mass
wc = ws if (args.sys=='A1' or args.sys=='B' or args.sys=='C') else 25*ws # Ohmic bath cutoff frequency
gam = 1*ms*ws if (args.sys=='A1' or args.sys=='A2' or args.sys=='B' or args.sys=='C') else 32*ms*ws # Ohmic bath friction coefficient
f = 12 if (args.sys=='A1' or args.sys=='B' or args.sys=='C') else 128 # Number of harmonic bath modes
tauL = gam/(ws**2*ms) # Debye longitudinal relaxation time for Zusman rate later
wj, cj, mj = gen_bath_params_conventional({"gam": gam, "wc": wc, "m": M, "f": f}) # Ohmic spectral density bath parameters

sys = 'A' if (args.sys=='A1' or args.sys=='A2' or args.sys=='A3') else 'B' if args.sys=='B' else 'C' # System type to use in Libra model system
K0 = 10**(args.logK)/beta # Diabatic coupling constant/prefactor
bq = 0.0 if (args.sys=='A1' or args.sys=='A2' or args.sys=='A3') else 3.0 # Diabatic coupling q coordinate exponential dependence
aq = 0.0 if (args.sys=='A1' or args.sys=='A2' or args.sys=='A3') else 8.0 if args.sys=='B' else 6.0 # q coordinate morse exponential parameter 
mq = 5.0e4 # q coordinate mass
wq = 5.0e-4 # q coordinate frequency
(q0, leps, Ea) = (2.1, args.leps/beta, 6.65e-3) # System C q coordinate double well parameters
(Aq, Bq, Cq) = get_ABC(q0, leps, Ea) # Numerically computing Aq, Bq, Cq from q0, leps, Ea
qhw = 1.0 # Hard wall potential location
khw = 1.0e5 # Hard wall potential strength

###########################################################################################
# ======= NOW CALLING ON TST CODE TO COMPUTE FREE ENERGIES AND KC-RPMD PARAMETERS ======= #
###########################################################################################

# Computing full fermi-golden rule rate saving to _sys_*/tst_data (no hardwall influence)
os.makedirs(pref + "/tst_data", exist_ok=True)

# Defining Kq and Vq functions of system for TST code
if sys == 'A':
    Vq = lambda q: 0.5*mq*wq**2*q**2
elif sys == 'B':
    Vq = lambda q: np.piecewise(q,[q >= 0.,q < 0.],
                                [lambda q: 0.5*mq*wq**2*q**2,
                                 lambda q: 0.5*mq*wq**2*(np.expm1(-aq*q)/aq)**2])
elif sys == 'C':
    Vq = lambda q: np.piecewise(q,[q >= 0.,q < 0.],
                                [lambda q: Aq*q**4 + Bq*q**3 + Cq*q**2,
                                 lambda q: Cq*(np.expm1(-aq*q)/aq)**2])

# Instantiation TST code object, computing eta, my, and gammay
qmin = -1.9
qmax = 3.0
kcrpmd_tst = KCRPMD_TST(beta, a, b, c, ms, ws, s0, s1, eps, K0, bq, Vq, qmin, qmax)

ydag = kcrpmd_tst.ydag
sdag = kcrpmd_tst.sdag

# IMPORTANT! without the hardwall separation we use eta, my, gammay computed for the whole system without hardwall potential.
# with the hard wall (sys C), instead eta, my, and gammay are computed from the right side lower well with the smaller diabatic coupling.
if args.hw != 0:
    print('Recomputing KC-RPMD parameters for system C using adiabatic well:')
    qmin_cp = kcrpmd_tst.qmin
    kcrpmd_tst.qmin = qhw
    kcrpmd_tst.set_eta_my_gammay()
    kcrpmd_tst.qmin = qmin_cp

# This is a cheap and dirty fix to recover original KC-RPMD using new KC-RPMD formalism.
if args.meth == 'ori':
    kcrpmd_tst.eta = 2*kcrpmd_tst.eta - np.sqrt(np.pi/kcrpmd_tst.a)
    kcrpmd_tst.a = 2*kcrpmd_tst.a
    kcrpmd_tst.c = 0.0

# Computing ground state free energy and KC-RPMD free energy (fully integrated)
FBO_full = kcrpmd_tst.FBO()[0]; FKC_full = kcrpmd_tst.FKC()[0]; kcrpmd_tst.reset()

# Now we add hardwall potential restriction to evaluate free energies and rates of each half separately (sys C only)
if args.hw == -1:
    kcrpmd_tst.qmin = qhw
elif args.hw == 1:
    kcrpmd_tst.qmax = qhw

# Coordinate arrays used to generate probability distributions for comparisions
# logfocus concentrates points around the center mu, used to resolve sharp kinetic constraint in plots
def logfocus(xmin, xmax, xpts, mu, sig, aexp):
    dx = (xmax - xmin)/(xpts - 1)
    uhop = sig + 2*np.expm1(-0.5*aexp*sig)/aexp
    umin = xmin - 0.5*uhop
    umax = xmax + 0.5*uhop
    upts = xpts + int(np.ceil(uhop/dx))
    u_ar = np.linspace(umin, umax, upts)
    x_ar = np.piecewise(u_ar,
                       [abs(u_ar - mu) <= 0.5*sig,
                        abs(u_ar - mu) > 0.5*sig],
                       [lambda u: mu + np.exp(-0.5*aexp*sig)/aexp*np.sign(u - mu)*(np.expm1(aexp*abs(u - mu))),
                        lambda u: u - np.sign(u - mu)*(0.5*sig + np.expm1(-0.5*aexp*sig)/aexp)])
    return x_ar
s_ar = logfocus(-7.0,7.0,1111,sdag,5.0,5.0)
q_ar = np.linspace(kcrpmd_tst.qmin,kcrpmd_tst.qmax,1357)
y_ar = np.linspace(-1.6,1.6,999)

# Now we calculate absolutely everything that might be useful later
# Hardwall influence if it's turned on, sys C only
if args.meth == 'adi':
    Phw = np.exp(-beta*(kcrpmd_tst.FBO()[0] - FBO_full)) # Hardwall potential probability from 0 to 1
    Pq_sdag = kcrpmd_tst.PBOq_s(sdag, q_ar) # Born-Oppenheimer probability along q at sdagger
    kGR = kcrpmd_tst.kGR()[0] # Fermi-golden rule rate
    kZUS = kcrpmd_tst.kZUS(tauL)[0] # Zusman rate
    kBOs = kcrpmd_tst.kBOs()[0] # Born-Oppenheimer TST rate
    np.savetxt(pref + "/tst_data/Pq_sdag.txt", np.column_stack((q_ar, Pq_sdag)))
    np.savetxt(pref + "/tst_data/kGR.txt", [kGR])
    np.savetxt(pref + "/tst_data/kZUS.txt", [kZUS])
    np.savetxt(pref + "/tst_data/kBOs.txt", [kBOs])
    np.savetxt(pref + "/tst_data/Phw.txt", [Phw])
elif args.meth == 'ori' or args.meth == 'new':
    Phw = np.exp(-beta*(kcrpmd_tst.FKC()[0] - FKC_full)) # Hardwall potential probability from 0 to 1
    if args.fix == "y":
        Ps_ydag = kcrpmd_tst.PKCs_y(ydag, s_ar) # KC-RPMD probability along s at ydagger
        Pq_ydag = kcrpmd_tst.PKCq_y(ydag, q_ar) # KC-RPMD probability along q at ydagger
        kKCy = kcrpmd_tst.kKCy()[0] # KC-RPMD TST rate along y
        np.savetxt(pref + "/tst_data/Ps_ydag.txt", np.column_stack((s_ar, Ps_ydag)))
        np.savetxt(pref + "/tst_data/Pq_ydag.txt", np.column_stack((q_ar, Pq_ydag)))
        np.savetxt(pref + "/tst_data/kKCy.txt", [kKCy])
        np.savetxt(pref + "/tst_data/Phw.txt", [Phw])
    elif args.fix == "s":
        Py_sdag = kcrpmd_tst.PKCy_s(y_ar, sdag) # KC-RPMD probability along y at sdagger
        Pq_sdag = kcrpmd_tst.PKCq_s(sdag, q_ar) # KC-RPMD probability along q at sdagger
        kKCs = kcrpmd_tst.kKCs()[0] # KC-RPMD TST rate along s
        np.savetxt(pref + "/tst_data/Py_sdag.txt", np.column_stack((y_ar, Py_sdag)))
        np.savetxt(pref + "/tst_data/Pq_sdag.txt", np.column_stack((q_ar, Pq_sdag)))
        np.savetxt(pref + "/tst_data/kKCs.txt", [kKCs])
        np.savetxt(pref + "/tst_data/Phw.txt", [Phw])

######################################################################################################################
# ======= NOW CREATING AND SAVING LIBRA CONTROL PARAMETERS FOR THERMALIZATION (PART 2) AND DYNAMICS (PART 3) ======= #
######################################################################################################################
nstates = 2
ndia = 2
nadi = 2
# setting ndof to 2 just for thermalization, bath is added later for dynamical trajectories
ndof = 2 #; ndof = 2 + len(mj)
ntraj = 1

# Save model parameters
_model_params = {"ms":ms, "ws":ws, "s0":s0, "s1":s1, "eps":eps,
                 "wj":wj, "cj":cj, "Mj":mj, "K0":K0,
                 "bq":bq, "aq":aq, "sys_type":sys, "mq":mq, "wq":wq,
                 "Aq":Aq, "Bq":Bq, "Cq":Cq,
                 "hard_wall":args.hw, "qhw":qhw, "khw":khw,
                 "model":1, "model0":1, "nstates": nstates}

with open(pref +  "/_model_params.txt", "w") as f:
    f.write(str(_model_params))

# Default parameters for thermalization calculation (dt, nsteps, and nprint will change for dynamics)
dyn_params = {"dt":41.34, "num_electronic_substeps":1, "nsteps":50000000, "nprint":1000,
              "prefix":"libra_data", "prefix2":"libra_data",
              "hdf5_output_level":-1, "mem_output_level":3, "txt_output_level":-1,
              "use_compression":0, "compression_level":[0,0,0], "progress_frequency":0.05,
              "ntraj":ntraj, "nstates":nstates}

# General Adiabatic recipe
def load_adiabatic(dyn_general, temp):
    dyn_general.update({"rep_tdse":1}) # adiabatic representation, wfc
    dyn_general.update({"ham_update_method":1})  # recompute only diabatic Hamiltonian
    dyn_general.update({"ham_transform_method":1}) # diabatic->adiabatic according to internal diagonalization
    dyn_general.update({"rep_force":1} ) # adiabatic
    dyn_general.update({"force_method":1} ) # state-specific  as in the TSH or adiabatic
    dyn_general.update({"time_overlap_method":1}) # explicitly compute it from the wavefunction info
    dyn_general.update({"isNBRA":0}) # no NBRA - Hamiltonians for all trajectories are computed explicitly [ default ]
    dyn_general.update({"Temperature":temp}) #Temperature of the system [ default ]
    dyn_general.update({"electronic_integrator":-1}) # No propagation

# KC-RPMD recipe
def load_kcrpmd(dyn_general, kcrpmd_tst, temp):
    dyn_general.update({"rep_tdse":0}) #diabatic representation, wfc
    dyn_general.update({"ham_update_method":1})  # recompute only diabatic Hamiltonian
    dyn_general.update({"ham_transform_method":0}) # don't do any transforms
    dyn_general.update({"rep_force":0}) # diabatic
    dyn_general.update({"force_method":4}) # KC-RPMD force
    dyn_general.update({"time_overlap_method":1}) # explicitly compute it from the wavefunction info
    dyn_general.update({"use_kcrpmd":1}) # use it 
    dyn_general.update({"kcrpmd_a":kcrpmd_tst.a})
    dyn_general.update({"kcrpmd_b":kcrpmd_tst.b})
    dyn_general.update({"kcrpmd_c":kcrpmd_tst.c})
    dyn_general.update({"kcrpmd_eta":kcrpmd_tst.eta})
    dyn_general.update({"kcrpmd_gamma":kcrpmd_tst.gammay})
    dyn_general.update({"kcrpmd_gammaKP":0.0})
    dyn_general.update({"kcrpmd_my":kcrpmd_tst.my})
    dyn_general.update({"isNBRA":0}) # no NBRA - Hamiltonians for all trajectories are computed explicitly [ default ]
    dyn_general.update({"Temperature":temp}) #Temperature of the system
    dyn_general.update({"electronic_integrator":-1}) # No propagation

# Langevin thermostat for nuclei
def load_langevin(dyn_general, _ndof):
    dyn_general.update({"ensemble":1}) #NVT
    dyn_general.update({"thermostat_params":{ "thermostat_type":"Langevin"
                                             , "Temperature":dyn_general["Temperature"]
                                             , "nu_therm":0.003 }})
    dyn_general.update({"thermostat_dofs":list(range(_ndof))})

# Load in control parameter recipe
if args.meth == 'adi':
    load_adiabatic(dyn_params, T)
    dyn_params.update({"properties_to_save":["timestep","time","q","p","f","Epot_ave","Ekin_ave","Etot_ave"]})
elif (args.meth == 'ori' or args.meth == 'new'):
    load_kcrpmd(dyn_params, kcrpmd_tst, T)
    dyn_params.update({"properties_to_save":["timestep","time","q","p","f","Epot_ave","Ekin_ave","Etot_ave", "y_aux_var","p_aux_var", "f_aux_var", "ekin_aux_var"]})

# Save control parameters for dynamics as copy before setting thermostat and constraints
_control_params_dynamics = dyn_params.copy()
_control_params_dynamics["dt"] = 0.08268
_control_params_dynamics["nsteps"] = 125000
_control_params_dynamics["nprint"] = 500
with open(pref +  "/_control_params_dynamics.txt", "w") as f:
    f.write(str(_control_params_dynamics))

# Load Langevin thermostat for thermalization
load_langevin(dyn_params, ndof)
if args.fix == 's':
    dyn_params.update({"constrained_dofs":[0]})
    dyn_params.update({"quantum_dofs":[]})
    dyn_params["thermostat_dofs"].pop(0)
    if (args.meth == 'ori' or args.meth == 'new'):
        dyn_params.update({"kcrpmd_gamma":0.003})
        dyn_params.update({"kcrpmd_gammaKP":0.003})

# Save control parameters for thermalization as copy
_control_params_thermalization = dyn_params.copy()
with open(pref +  "/_control_params_thermalization.txt", "w") as f:
    f.write(str(_control_params_thermalization))

##########################################################################################
# ======= NOW CREATING AND SAVING INITIAL CONDITIONS FOR THERMALIZATION (PART 2) ======= #
##########################################################################################

# As discussed in the paper, free energies do not depend on the mass of coordinates.
# We choose masses so that the harmoic frequencies match the timestep.
# n_therm is redundancy factor, mass of s coordinate will either follow harmonic V0(s), or gaussian restraint for small beta*K0
n_therm = 1000
if sys == 'C' and args.hw == -1:
    Kref = K0*np.exp(-bq*2.1)
else:
    Kref = K0
lam = ms*ws**2*(s0-s1)**2/2
ms_therm = ms*max(ws, np.sqrt(4*a*lam/(beta*Kref**2))*ws)**2*(n_therm*41.34/(2*np.pi))**2
#ms_therm = max(ms*ws**2*(n_therm*41.34/(2*np.pi))**2, 2*a*beta*(n_therm*ms*ws**2*(s0-s1)*41.34/(2*np.pi*beta*Kref))**2)
mq_therm = mq*wq**2*(n_therm*41.34/(2*np.pi))**2
my_therm = 250000.0

# Initial conditions and masses of nuclear coordinates for thermalization
_nucl_params_therm = {"q":[sdag,qhw], "p":[0.0,0.0], "mass": [ms_therm,mq_therm],
                      "force_constant":[4*ms_therm/(beta**2),4*mq_therm/(beta**2)],
                      "init_type":1, "ntraj": ntraj, "ndof": 2}

# Initial conditions and mass of auxiliary coordinate for thermalization
_elec_params_therm = {"init_type":0, "nstates":nstates, "rep":1, "istate": 0,
                      "ntraj":ntraj, "ndia":ndia, "nadi":nadi}
if (args.meth == 'ori' or args.meth == 'new'):
    _elec_params_therm.update({"y_aux_var":[ydag]})
    _elec_params_therm.update({"p_aux_var":[0.0]})
    _elec_params_therm.update({"m_aux_var":[my_therm]})

# Saving initial conditions for thermalization
with open(pref +  "/_init_nucl_thermalization.txt", "w") as f:
    f.write(str(_nucl_params_therm))

with open(pref +  "/_init_elec_thermalization.txt", "w") as f:
    f.write(str(_elec_params_therm))

