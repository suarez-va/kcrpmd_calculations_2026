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

from kcrpmd_utils.Spin_Boson_Conventional import spin_boson_conventional

class tmp:
    pass

def get_ABC(q0, leps, Ea):
    def minimize_me(x):
        A, B, C = x
        A_abs, B_abs = np.abs(A), np.abs(B)

        discriminant = 9 * B**2 - 32 * A_abs * C
        if discriminant < 0:
            return 1e12  # large penalty to avoid invalid region

        delta = np.sqrt(discriminant)
        q_plus = (3 * B_abs + delta) / (8 * A_abs)
        q_minus = (3 * B_abs - delta) / (8 * A_abs)

        E_plus = A_abs * q_plus**4 - B_abs * q_plus**3 + C * q_plus**2
        E_minus = A_abs * q_minus**4 - B_abs * q_minus**3 + C * q_minus**2

        term_q0 = (q0 - q_plus)**2
        term_leps = (leps + E_plus)**2
        term_Ea = (Ea - E_minus)**2

        return term_q0 + term_leps + term_Ea

    x_guess = np.array([16 * Ea / q0**4, 32 * Ea / q0**3, 16 * Ea / q0**2])

    for _ in range(7):
        res = minimize(minimize_me, x_guess, method='Nelder-Mead', tol=1e-11)
        x_guess = res.x

    Aq, Bq, Cq = float(np.abs(x_guess[0])), float(-np.abs(x_guess[1])), float(x_guess[2])
    return Aq, Bq, Cq


def kcrpmd_system_bath(q, params, full_id):
    """

    effective solvent coordinate factored 2-state spin-boson model as defined in KC-RPMD paper

         | 0.5*ms*ws^2*(s-s0)^2            K0*exp(-bq*q)       |
    H =  |                                                     | + I*[Vb(s,x) + Vda(q)]
         |    K0*exp(-bq*q)         0.5*ms*ws^2*(s-s1)^2 + eps |


    Vb(s,x) = sumj{0.5*Mj*wj^2*(xj-cj*s/(Mj*wj^2))^2}

    Ohmic spectral density as defined in KC-RPMD paper
    J(w) = gam*w*exp(-w/wc)

    Args:
        q ( MATRIX(ndof, 1) ): coordinates of the particle, ndof = f + 2
        params ( dictionary ): model parameters
            * **params["ms"]** ( double ): s coordinate mass [ default: 1836., units: a.u. ]
            * **params["ws"]** ( double ): s coordinate angular frequency [ default: 2.28e-3, units: a.u. ]
            * **params["s0"]** ( double ): V0 parabola center [ default: -2.40, units: a.u. ]
            * **params["s1"]** ( double ): V1 parabola center [ default: 2.40, units: a.u. ]
            * **params["eps"]** ( double ): V1 parabola vertical shift [ default: 0., units: a.u. ]
            * **params["wj"]** (list of f doubles): frequencies [ default: [], units: a.u.]
            * **params["cj"]** (list of f doubles): diagonal linear couplings [ default: [], units: a.u.]
            * **params["Mj"]** (list of f doubles): masses of nuclear DOFS [ default: [], units: a.u.]
            * **params["K0"]** ( double ): electronic coupling strength [ default: 6.67e-7, units: a.u. ]
            * **params["bq"]** ( double ): exponential coupling parameter [ default: 0., units: a.u. ]
            * **params["aq"]** ( double ): systems B and C morse potential exponential prefactor [ default: 5.0, units: a.u. ]
            * **params["sys_type"]** ( str ): Different options for Vda(q):
              - A: system A of KC-RPMD paper
              - B: system B of KC-RPMD paper
              - C: system C of KC-RPMD paper
            * **params["mq"]** ( double ): q coordinate mass [ default: 5.00e4, units: a.u. ]
            * **params["wq"]** ( double ): systems A and B q coordinate angular frequency [ default: 5.00e-4, units: a.u. ]
            * **params["Aq"]** ( double ): system C quartic coefficient [ default: 1.041e-2, units: a.u. ]
            * **params["Bq"]** ( double ): system C cubic coefficient [ default: 4.065e-2, units: a.u. ]
            * **params["Cq"]** ( double ): system C quadratic coefficient [ default: 3.622e-2, units: a.u. ]
            * **params["hard_wall"]** ( int ): whether to set a sextic hard wall potential for Vda(q):
              - -1: left side hard wall
              - 0: no hard wall [ default ]
              - 1: right side hard wall
            * **params["qhw"]** ( double ): hard wall position [ default: 1.00, units: a.u. ]
            * **params["khw"]** ( double ): hard wall constant [ default: 1.00e5, units: a.u. ]

    Returns:
        PyObject: obj, with the members:

            * obj.ham_dia ( CMATRIX(2,2) ): diabatic Hamiltonian
            * obj.ovlp_dia ( CMATRIX(2,2) ): overlap of the basis (diabatic) states [ identity ]
            * obj.d1ham_dia ( list of ndof CMATRIX(2,2) objects ):
                derivatives of the diabatic Hamiltonian w.r.t. the nuclear coordinate
            * obj.dc1_dia ( list of ndof CMATRIX(2,2) objects ): derivative coupling in the diabatic basis [ zero ]
 
    """

    critical_params = []
    default_params = {"ms":1836.0, "ws":2.28e-3, "s0":-2.4, "s1":2.4, "eps":0.0,
                      "wj":[], "cj":[], "Mj":[], "K0":6.67e-7,
                      "bq":0.0, "aq":5.0, "sys_type":'A', "mq":5e4, "wq":5e-4,
                      "Aq":1.041e-2, "Bq":-4.065e-2, "Cq":3.622e-2,
                      "hard_wall":0, "qhw":1., "khw":1e5}
    comn.check_input(params, default_params, critical_params)

    ms = params["ms"]
    ws = params["ws"]
    s0 = params["s0"]
    s1 = params["s1"]
    eps = params["eps"]
    wj = params["wj"]
    cj = params["cj"]
    Mj = params["Mj"]
    K0 = params["K0"]
    bq = params["bq"]
    aq = params["aq"]
    sys_type = params["sys_type"]
    mq = params["mq"]
    wq = params["wq"]
    Aq = params["Aq"]
    Bq = params["Bq"]
    Cq = params["Cq"]
    hard_wall = params["hard_wall"]
    qhw = params["qhw"]
    khw = params["khw"]

    ndof = q.num_of_rows  # the number of nuclear DOFs

    indx = 0
    if full_id !=None:
        Id = Cpp2Py(full_id)
        indx = Id[-1]

    # handeling s and x coordinates
    spin_boson_params = {"mass0": ms, "omega0": ws, "Lambda": 0.5*ms*ws**2*(s0 - s1)**2,
                         "Delta": K0, "epsilon": eps, "omega": wj, "coupl": cj, "mass": Mj}

    qm1 = MATRIX(ndof - 1, 1)
    for i in range(ndof - 1):
        qm1.set(i,0,q.get(i,indx))

    obj = spin_boson_conventional(qm1, spin_boson_params, full_id)

    obj.d1ham_dia.append(CMATRIX(2,2))
    obj.dc1_dia.append(CMATRIX(2,2))

    #=========== Energies & Derivatives (q coordinate only) ===============
    qda = q.get(ndof - 1,indx)

    Kq = K0*np.exp(-bq*qda)
    dKq = -bq*K0*np.exp(-bq*qda)

    Vq = 0.0
    dVq = 0.0
    if (sys_type == 'A'):
        Vq = 0.5*mq*wq**2*qda**2
        dVq = mq*wq**2*qda
    elif (sys_type == 'B'):
        if (qda > 0.0 or aq == 0.0):
            Vq = 0.5*mq*wq**2*qda**2
            dVq = mq*wq**2*qda
        else:
            #print(f'EXP STUFF: {qda**2} vs {(np.expm1(-aq*qda)/aq)**2}, diff: {(np.expm1(-aq*qda)/aq)**2 - qda**2}')
            Vq = 0.5*mq*wq**2*(np.expm1(-aq*qda)/aq)**2
            dVq = -mq*wq**2*np.exp(-aq*qda)*np.expm1(-aq*qda)/aq
    elif (sys_type == 'C'):
        if (qda > 0.0):
            Vq = Aq*qda**4 + Bq*qda**3 + Cq*qda**2
            dVq = 4*Aq*qda**3 + 3*Bq*qda**2 + 2*Cq*qda
        elif (qda <= 0.0 and aq != 0.0):
            Vq = Cq*(np.expm1(-aq*qda)/aq)**2
            dVq = -2*Cq*np.exp(-aq*qda)*np.expm1(-aq*qda)/aq
        else:
            Vq = Cq*qda**2
            dVq = 2*Cq*qda

    if ((hard_wall == -1 and qda <= qhw) or (hard_wall == 1 and qda >= qhw)):
        Vq += khw*(qda - qhw)**6
        dVq += 6*khw*(qda - qhw)**5

    obj.ham_dia.add(0,0,Vq*(1.0 + 0.0j)); obj.ham_dia.set(0,1,Kq*(1.0 + 0.0j))
    obj.ham_dia.set(1,0,Kq*(1.0 + 0.0j)); obj.ham_dia.add(1,1,Vq*(1.0 + 0.0j))

    obj.d1ham_dia[ndof - 1].add(0,0,dVq*(1.0 + 0.0j)); obj.d1ham_dia[ndof - 1].set(0,1,dKq*(1.0 + 0.0j))
    obj.d1ham_dia[ndof - 1].set(1,0,dKq*(1.0 + 0.0j)); obj.d1ham_dia[ndof - 1].add(1,1,dVq*(1.0 + 0.0j))

    return obj


def junk_test(q, _params, full_id):
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
    if ndof != len(omega)+2:
        print(f"Shape of q coordinates inconsistent with bath parameters. q has {ndof} dof, omega should be length {ndof-1}\nExiting now...\n")
        sys.exit(0)

    indx = 0
    if full_id !=None:
        Id = Cpp2Py(full_id)
        indx = Id[-1]

    qm1 = MATRIX(ndof - 1, 1)
    for i in range(ndof - 1):
        qm1.set(i,0,q.get(i,indx))

    obj = spin_boson_conventional(qm1, _params, full_id)

    obj.d1ham_dia.append(CMATRIX(2,2))
    obj.dc1_dia.append(CMATRIX(2,2))


    qda = q.get(ndof,indx)
    print(qda)
    obj.ham_dia.set(0,1,Delta*(1.0 + 0.0j))
    obj.ham_dia.set(1,0,Delta*(1.0 + 0.0j))
    obj.d1ham_dia[ndof - 1].set(0,1,(0.0 + 0.0j))
    obj.d1ham_dia[ndof - 1].set(1,0,(0.0 + 0.0j))

    mq = 50000
    wq = 2.2e-3
    obj.ham_dia.add(0,0,0.5*mq*wq**2*qda**2*(1.0 + 0.0j))
    obj.ham_dia.add(1,1,0.5*mq*wq**2*qda**2*(1.0 + 0.0j))
    obj.d1ham_dia[ndof - 1].add(0,0,mq*wq**2*qda*(1.0 + 0.0j))
    obj.d1ham_dia[ndof - 1].add(1,1,mq*wq**2*qda*(1.0 + 0.0j))
    #print(len(obj.d1ham_dia))

    return obj

