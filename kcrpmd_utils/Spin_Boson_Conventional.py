import os
import sys
import math
import copy

if sys.platform == "cygwin":
    from cyglibra_core import *
elif sys.platform == "linux" or sys.platform == "linux2":
    from liblibra_core import *
import util.libutil as comn
import libra_py.units as units

import numpy as np
from scipy.optimize import minimize

class tmp:
    pass

def gen_bath_params_conventional(_params):
    """
    Generates the parameters for Ohmic bath coupled to nuclear degree of freedom.
    (discretization according to: https://doi.org/10.1063/1.1850093)

    Hamiltonian form:
    H(vec{q},vec{q}) = p_0^2/(2*m_0) + V_1(q_0) + sum_{i=1}^{f}{ p_i^2/(2*m) + 1/2*m*w_i^2*(q_i - c_i*q_0/(m*w_i^2))^2 }

    Ohmic spectral density:
    J(w) = gam * w * exp(-w / wc)

    Args:
        params ( dictionary ): model parameters, can contain:

            * **params["gam"]** ( double ): Ohmic bath coupling strength [ default: 1.0, units: mass/time^2 ]
            * **params["wc"]** ( double ): Ohmic bath cutoff frequency [ default: 1.0, units: 1/time ]
            * **params["m"]** ( double ): Ohmic bath mass(es) [ default: 1.0, units: mass ]
            * **params["f"]** ( integer ): number of Ohmic bath modes [ default: 1 ]

    Returns:
        tuple: (list, list, list)

            * omega ( list of `f` doubles ): bath frequencies [ units: 1/time ]
            * coupl ( list of `f` doubles ): coupling strengths [ units: mass/time^2 ]
            * mass ( list of `f` doubles): masses of oscillators [ units: mass ]

    """

    params = dict(_params)

    critical_params = []
    default_params = {"gam": 1.0, "wc": 1.0, "m": 1.0, "f": 1}
    comn.check_input(params, default_params, critical_params)

    gam = params["gam"]
    wc = params["wc"]
    m = params["m"]
    f = params["f"]

    omega, coupl, mass = [], [], []

    pref = math.sqrt(2*gam*m*wc/(f*math.pi))
    omega = [-wc*math.log((i + 0.5)/f) for i in range(f)]
    coupl = [omega[i]*pref for i in range(f)]
    mass = [m for i in range(f)]

    return omega, coupl, mass


def spin_boson_conventional(q, _params, full_id):
    """
    2-state conventional spin-boson model of the form:

        | 1/2*m_0*w_0^2*(q_0 + sqrt(Lam/(2*m_0*w_0^2)))^2                                   Del                        |
    V = |                                                                                                              | + I*Vb(vec{q})
        |                       Del                              1/2*m_0*w_0^2*(q_0 + sqrt(Lam/(2*m_0*w_0^2)))^2 - eps |

    Vb(vec{q}) = sum_{i}{ 1/2*m_i*w_i^2*(q_i - c_i*q_0/(m_i*w_i^2))^2 }

    Args:
        q ( MATRIX(ndof, 1) ): coordinates of the particle, ndof = f + 2
        params ( dictionary ): model parameters
            * **params["mass0"]** ( double ): m_0 mass of coordinate q_0 [ default: 1.0, units: mass ]
            * **params["omega0"]** ( double ): w_0 frequency of coordinate q_0 [ default: 1.0, units: 1/time ]
            * **params["Lambda"]** ( double ): Lam reorganization energy [ default: 1.0, units: energy ]
            * **params["Delta"]** ( double ): Del diabatic coupling strength [ default: 1.0, units: energy ]
            * **params["epsilon"]** ( double ): eps driving force [ default: 0.0, units: energy ]

            * **params["omega"]** (list of doubles): w_i bath frequencies [ default: [], units: 1/time ]
            * **params["coupl"]** (list of doubles): c_i bath coupling coefficients [ default: [], units: mass/time^2]
            * **params["mass"]** (list of doubles): m_i bath masses [ default: [], units: mass]

    Returns:
        PyObject: obj, with the members:

            * obj.ham_dia ( CMATRIX(2,2) ): diabatic Hamiltonian
            * obj.ovlp_dia ( CMATRIX(2,2) ): overlap of the basis (diabatic) states [ identity ]
            * obj.d1ham_dia ( list of ndof CMATRIX(2,2) objects ):
                derivatives of the diabatic Hamiltonian w.r.t. the nuclear coordinate
            * obj.dc1_dia ( list of ndof CMATRIX(2,2) objects ): derivative coupling in the diabatic basis [ zero ]
 
    """

    params = dict(_params)

    critical_params = []
    default_params = {"mass0":1.0, "omega0":1.0, "Lambda":1.0, "Delta":1.0, "epsilon":0.0,
                      "omega":[], "coupl":[], "mass":[]}
    comn.check_input(params, default_params, critical_params)

    mass0 = params["mass0"]
    omega0 = params["omega0"]
    Lambda = params["Lambda"]
    Delta = params["Delta"]
    epsilon = params["epsilon"]

    omega = params["omega"]
    coupl = params["coupl"]
    mass = params["mass"]

    ndof = q.num_of_rows  # the number of nuclear DOFs
    if ndof != len(omega)+1:
        print(f"Shape of q coordinates inconsistent with bath parameters. q has {ndof} dof, omega should be length {ndof-1}\nExiting now...\n")
        sys.exit(0)

    obj = tmp()
    obj.ham_dia = CMATRIX(2,2)
    obj.ovlp_dia = CMATRIX(2,2);  obj.ovlp_dia.identity()
    obj.d1ham_dia = CMATRIXList()
    obj.dc1_dia = CMATRIXList()

    for i in range(ndof):
        obj.d1ham_dia.append(CMATRIX(2,2))
        obj.dc1_dia.append(CMATRIX(2,2))

    indx = 0
    if full_id !=None:
        Id = Cpp2Py(full_id)
        indx = Id[-1]

    if indx != 0:
        print(f'is it this?: {indx}')

    #=========== Energies & Derivatives ===============

    # q0 is the effective solvent coordinate, unique from other bath coordinates qi {i>0}
    q0 = q.get(0,indx)

    # energy from just q0 coordinate without bath qi {i>0}
    obj.ham_dia.set(0,0,(0.5*mass0*omega0**2*(q0 + math.sqrt(Lambda/(2*mass0*omega0**2)))**2)*(1.0 + 0.0j))
    obj.ham_dia.set(1,1,(0.5*mass0*omega0**2*(q0 - math.sqrt(Lambda/(2*mass0*omega0**2)))**2 - epsilon)*(1.0 + 0.0j))
    obj.ham_dia.set(0,1,Delta*(1.0 + 0.0j))
    obj.ham_dia.set(1,0,Delta*(1.0 + 0.0j))

    # derivative w.r.t. q0 without bath coordinates qi {i>0}
    obj.d1ham_dia[0].add(0,0,(mass0*omega0**2*(q0 + math.sqrt(Lambda/(2*mass0*omega0**2))))*(1.0 + 0.0j))
    obj.d1ham_dia[0].add(1,1,(mass0*omega0**2*(q0 - math.sqrt(Lambda/(2*mass0*omega0**2))))*(1.0 + 0.0j))

    x = 0.0
    for i in range(1,ndof):
        qi = q.get(i,indx)

        ci = coupl[i - 1]
        ki = mass[i - 1]*omega[i - 1]**2
        ui = qi - ci*q0/ki

        x += 0.5*ki*ui**2
        y = -ci*ui
        z = ki*ui

        # derivative w.r.t. q0 with bath coordinates qi {i>0}
        obj.d1ham_dia[0].add(0,0,y*(1.0 + 0.0j))
        obj.d1ham_dia[0].add(1,1,y*(1.0 + 0.0j))
        # derivative w.r.t. bath coordinates qi {i>0}:
        obj.d1ham_dia[i].add(0,0,z*(1.0 + 0.0j))
        obj.d1ham_dia[i].add(1,1,z*(1.0 + 0.0j))

    # energy from coordinate q0 coupling to bath coordinates qi {i>0}
    obj.ham_dia.add(0,0,x*(1.0 + 0.0j))
    obj.ham_dia.add(1,1,x*(1.0 + 0.0j))

    return obj

