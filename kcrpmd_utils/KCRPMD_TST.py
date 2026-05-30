import os
import sys
import math
import copy
import time

if sys.platform == "cygwin":
    from cyglibra_core import *
elif sys.platform == "linux" or sys.platform == "linux2":
    from liblibra_core import *
import util.libutil as comn
import libra_py.units as units

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


from scipy.optimize import minimize



class KCRPMD_TST:

    def __init__(self, beta, a, b, c, ms, ws, s0, s1, eps, K0, bq, Vq, qmin, qmax):

        """
        effective solvent coordinate factored 2-state spin-boson model as defined in KC-RPMD paper

             | 0.5*ms*ws^2*(s-s0)^2           K0*e^(-bq*q)        |
        H =  |                                                    | + I*[Vq(q) + Vb(s,x)]
             |     K0*e^(-bq*q)        0.5*ms*ws^2*(s-s1)^2 + eps |

        Vb(s,vec{x}) = sum_{j}{ 1/2*m_j*w_j^2*(x_j - c_j*s/(m_j*w_j^2))^2 }

        Args:
            beta ( double ): boltzmann factor [ units: a.u. ]
            a ( double ): KC-RPMD kinetic constraint parameter a [ units: a.u. ]
            b ( double ): KC-RPMD heavy-side parameter b [ units: a.u. ]
            c ( double ): KC-RPMD kinetic constraint switching parameter c [ units: a.u. ]
            ms ( double ): s coordinate mass [ units: a.u. ]
            ws ( double ): s coordinate angular frequency [ units: a.u. ]
            s0 ( double ): V0 parabola center [ units: a.u. ]
            s1 ( double ): V1 parabola center [ units: a.u. ]
            eps ( double ): V1 parabola vertical shift [ units: a.u. ]
            K0 ( double ): diabatic coulping strength parameter [ units: a.u. ]
            bq ( double ): diabatic coupling q coordinate exponential prefactor [ units: a.u. ]
            Vq ( vectorized function ): q coordinate potential [ units: a.u. ]
            qmin ( double ): minimum value to consider for integration along q [ units: a.u. ]
            qmax ( double ): maximum value to consider for integration along q [ units: a.u. ]
            #wj (list of doubles): w_j bath frequencies [ default: [], units: a.u. ]
            #cj (list of doubles): c_j bath coupling coefficients [ default: [], units: a.u. ]
            #mj (list of doubles): m_j bath masses [ default: [], units: a.u. ]
        """
        self.beta = beta
        self.a = a
        self.b = b
        self.c = c
        self.ms = ms
        self.ws = ws
        self.s0 = s0
        self.s1 = s1
        self.eps = eps
        self.K0 = K0
        self.bq = bq
        self.Vq = Vq
        self.qmin = qmin
        self.qmax = qmax

        self.ydag = 0.0
        self.sdag = 0.5*(self.s0 + self.s1) - self.eps / (self.ms*self.ws**2*(self.s0 - self.s1))
        self.lam = 0.5*self.ms*self.ws**2*(self.s0 - self.s1)**2

        self.V0s = lambda s: 0.5*self.ms*self.ws**2*(s - self.s0)**2
        self.V1s = lambda s: 0.5*self.ms*self.ws**2*(s - self.s1)**2 - self.eps
        self.Vs = lambda s: 0.5*(self.ms*self.ws**2*((0.5*(self.s0 + self.s1) - s)**2 + 0.25*(self.s0 - self.s1)**2) - self.eps)
        self.Ds = lambda s: 0.5*(self.ms*self.ws**2*(self.s0 - self.s1)*(0.5*(self.s0 + self.s1) - s) + self.eps)
        self.Kq = lambda q: abs(self.K0*np.exp(-self.bq*q))

        self.xatol = 1e-7
        self.epsabs=1.0e-10
        self.epsrel=1.0e-13
        self.limit=1000
        self.stds = 1/np.sqrt(self.beta*self.ms*self.ws**2)
        self.smin = self.s0 - 10*self.stds
        self.smax = self.s1 + 10*self.stds
        self.ymin = -1.6
        self.ymax = 1.6

        self.set_eta_my_gammay()
        self.lnSx = lambda x, A, B: np.piecewise(x,
                                                 [x >= B,
                                                  x < B],
                                                 [lambda x: -2*A*(x - B) - np.log(1 + np.exp(-2*A*(x - B))),
                                                  lambda x: -np.log(1 + np.exp(2*A*(x - B)))])

        self.Aq = lambda q: self.a*np.exp(self.lnSx(self.beta*self.Kq(q), self.c, 1.0))
        self.Cq = lambda q: (1 + (np.sqrt(self.a/np.pi)*self.eta
                                  *np.exp(0.5*self.lnSx(self.beta*self.Kq(q), self.c, 1.0)) - 1)
                                 *np.exp(self.lnSx(self.beta*self.Kq(q), self.c, 1.0)))
        self.Vr = lambda theta, y: -1/self.beta*self.lnSx(abs(y - theta), self.b, 0.5)


    def reset(self):
        if all(hasattr(self, name) for name in ["FBOavg", "FBOerr"]):
            del self.FBOavg; del self.FBOerr
        if all(hasattr(self, name) for name in ["F0avg", "F0err"]):
            del self.F0avg; del self.F0err
        if all(hasattr(self, name) for name in ["FKPavg", "FKPerr"]):
            del self.FKPavg; del self.FKPerr
        if all(hasattr(self, name) for name in ["F1avg", "F1err"]):
            del self.F1avg; del self.F1err
        if all(hasattr(self, name) for name in ["FKCavg", "FKCerr"]):
            del self.FKCavg; del self.FKCerr

    def minimize_Fx(self, Fx, xmin, xmax):
        xref = minimize_scalar(lambda x: self.beta*Fx(x), bounds=(xmin, xmax), method='bounded', options={'xatol': self.xatol}).x
        return xref

    def integrate_Fx(self, Fx, xref, points, xmin, xmax):
        Fref = Fx(xref)
        exp_fun = lambda x: np.exp(-self.beta*(Fx(x) - Fref))
        Fref += - 1/self.beta*np.log(quad(exp_fun, xmin, xmax, epsabs=0.0, epsrel=0.1, limit=self.limit, points=points)[0])
        val, err = quad(exp_fun, xmin, xmax, epsabs=self.beta*self.epsabs, epsrel=self.beta*abs(Fref)*self.epsrel, limit=self.limit, points=points)
        Fval = Fref - 1/self.beta*np.log(val)
        Ferr = abs(err/(self.beta*val))
        return Fval, Ferr

    def set_eta_my_gammay(self):
        Fq = lambda q: self.Vq(q)
        qref = self.minimize_Fx(Fq, self.qmin, self.qmax)
        points=[qref]
        Fval, Ferr = self.integrate_Fx(Fq, qref, points, self.qmin, self.qmax)
        Pq = lambda q: np.exp(-self.beta*(Fq(q) - Fval))

        KPq = lambda q: self.Kq(q)*Pq(q)
        K2Pq = lambda q: self.Kq(q)**2*Pq(q)
        K3Pq = lambda q: self.Kq(q)**3*Pq(q)
        lnKPq = lambda q: np.log(self.Kq(q))*Pq(q)

        Kavg, Kerr = quad(KPq, self.qmin, self.qmax, epsabs=0.0, epsrel=self.epsrel, limit=self.limit, points=points)
        K2avg, K2err = quad(K2Pq, self.qmin, self.qmax, epsabs=0.0, epsrel=self.epsrel, limit=self.limit, points=points)
        K3avg, K3err = quad(K3Pq, self.qmin, self.qmax, epsabs=0.0, epsrel=self.epsrel, limit=self.limit, points=points)
        lnKavg, lnKerr = quad(lnKPq, self.qmin, self.qmax, epsabs=0.0, epsrel=self.epsrel, limit=self.limit, points=points)

        self.eta = 2*np.pi*Kavg*K2avg/K3avg
        self.my = self.beta**3/(2*np.pi)*Kavg**2
        self.gammay = np.sqrt(np.pi/2)*np.sqrt(1 + np.abs(-2*np.log(np.sqrt(self.a/np.pi)*self.eta*self.beta**2) - 4*lnKavg))/(self.beta**2*Kavg)

        eta_err = self.eta*np.sqrt((Kerr/Kavg)**2 + (K2err/K2avg)**2 + (K3err/K3avg)**2)
        my_err = self.my*2*(Kerr/Kavg)
        gammay_err = self.gammay*np.sqrt((2*lnKerr/(1 + np.abs(-2*np.log(np.sqrt(self.a/np.pi)*self.eta*self.beta**2) - 4*lnKavg)))**2 + (Kerr/Kavg)**2)

        print(f'KC-RPMD TST: η = {self.eta}, δη = {eta_err}')
        print(f'KC-RPMD TST: m = {self.my}, δm = {my_err}')
        print(f'KC-RPMD TST: γ = {self.gammay}, δγ = {gammay_err}')

        return None


    ##################################
    ####### general potentials #######
    ##################################

    # Diabatic potential V0
    def V0(self, s, q):
        if np.isscalar(s) or np.isscalar(q):
            return self.V0s(s) + self.Vq(q)
        else:
            S,Q = np.meshgrid(s,q)
            return self.V0s(S) + self.Vq(Q)

    # Diabatic potential V1
    def V1(self, s, q):
        if np.isscalar(s) or np.isscalar(q):
            return self.V1s(s) + self.Vq(q)
        else:
            S,Q = np.meshgrid(s,q)
            return self.V1s(S) + self.Vq(Q)

    # State independent potential V
    def V(self, s, q):
        if np.isscalar(s) or np.isscalar(q):
            return self.Vs(s) + self.Vq(q)
        else:
            S,Q = np.meshgrid(s,q)
            return self.Vs(S) + self.Vq(Q)

    # Diabatic potential energy gap D=0.5*(V0 - V1)
    def D(self, s, q):
        if np.isscalar(q):
            return self.Ds(s)
        elif np.isscalar(s):
            return np.full_like(q,self.Ds(s))
        else:
            S,Q = np.meshgrid(s,q)
            return self.Ds(S)

    # Diabatic coupling energy
    def K(self, s, q):
        if np.isscalar(s):
            return self.Kq(q)
        elif np.isscalar(q):
            return abs(np.full_like(s,self.Kq(q)))
        else:
            S,Q = np.meshgrid(s,q)
            return self.Kq(Q)

    # Adiabatic state dependent potential Vz
    def Vz(self, s, q):
        return np.sqrt(self.D(s,q)**2 + self.K(s,q)**2)

    # KC-RPMD kinked pair potential Vkp (without state independent part V)
    def Vkp(self, s, q):
        K = self.K(s,q)
        D = self.D(s,q)
        Vz = np.sqrt(D**2 + K**2)

        Vtmp = (K**2/(Vz + abs(D))
                - 1/self.beta*np.log1p(-np.expm1(-2*self.beta*(K**2/(Vz + abs(D))))
                                       *np.exp(-2*self.beta*abs(D))
                                       /(1 + np.exp(-2*self.beta*Vz))))
        Vkp = (-Vz
               - 1/self.beta*np.log1p(np.exp(-2*self.beta*Vz))
               - 1/self.beta*np.log(-np.expm1(-self.beta*Vtmp)))

        return Vkp

    # KC-RPMD kinetic constraint penalty potential Vkc
    def Vkc(self, s, q):
        if np.isscalar(s) or np.isscalar(q):
            return (self.Aq(q)*(2*self.Ds(s)/self.Kq(q))**2 - np.log(self.Cq(q)))/self.beta
        else:
            S,Q = np.meshgrid(s,q)
            return (self.Aq(Q)*(2*self.Ds(S)/self.Kq(Q))**2 - np.log(self.Cq(Q)))/self.beta

    # Full adiabatic ground state Born-Oppenheimer potential VBO
    def VBO(self, s, q):
        return self.V(s,q) - self.Vz(s,q)

    # Full KC-RPMD efective potential VKC
    def VKC(self, theta, s, q):
        if theta == -1:
            return self.V0(s,q)
        elif theta == 0:
            return self.V(s,q) + self.Vkp(s,q) + self.Vkc(s,q)
        elif theta == 1:
            return self.V1(s,q)
        else:
            print('Invalid Theta!')
            return np.sqrt(-1)


    ##############################
    ####### free energies  #######
    ##############################

    # Born-Oppenheimer free energy along s
    def FBOs(self, s):
        if np.isscalar(s):
            Fq = lambda q: self.VBO(s,q)
            qref = self.minimize_Fx(Fq, self.qmin, self.qmax)
            points = [qref]
            Fval, Ferr = self.integrate_Fx(Fq, qref, points, self.qmin, self.qmax)
        else:
            Fval = np.zeros_like(s); Ferr = np.zeros_like(s)
            for i, si in enumerate(s):
                Fq = lambda q: self.VBO(si,q)
                qref = self.minimize_Fx(Fq, self.qmin, self.qmax)
                points = [qref]
                Fval[i], Ferr[i] = self.integrate_Fx(Fq, qref, points, self.qmin, self.qmax)
        return Fval, Ferr

    # Born-Oppenheimer free energy along q
    def FBOq(self, q):
        if np.isscalar(q):
            Fs = lambda s: self.VBO(s,q)
            sref = min([self.s0, self.sdag, self.s1], key=Fs)
            points = [sref]
            Fval, Ferr = self.integrate_Fx(Fs, sref, points, self.smin, self.smax)
        else:
            Fval = np.zeros_like(q); Ferr = np.zeros_like(q)
            for i, qi in enumerate(q):
                Fs = lambda s: self.VBO(s,qi)
                sref = min([self.s0, self.sdag, self.s1], key=Fs)
                points = [sref]
                Fval[i], Ferr[i] = self.integrate_Fx(Fs, sref, points, self.smin, self.smax)
        return Fval, Ferr

    # Born-Oppenheimer total free energy
    def FBO(self):
        Fs = lambda s: self.FBOs(s)[0]
        if not all(hasattr(self, name) for name in ["FBOavg", "FBOerr"]):
            sref = min([self.s0, self.sdag, self.s1], key=Fs)
            points = [sref]
            Fval, Ferr = self.integrate_Fx(Fs, sref, points, self.smin, self.smax)
            self.FBOavg, self.FBOerr = Fval, Ferr
        else:
             Fval, Ferr = self.FBOavg, self.FBOerr
        return Fval, Ferr

    # KC-RPMD free energy at theta along s
    def FKCthetas(self, theta, s):
        if np.isscalar(s):
            Fq = lambda q: self.VKC(theta,s,q)
            qref = self.minimize_Fx(Fq, self.qmin, self.qmax)
            if theta == 0:
                dqref = abs(minimize_scalar(lambda dq: (self.beta*(Fq(qref + dq) - Fq(qref)) - 0.5)**2, bounds=(self.qmin - qref, self.qmax - qref), method='bounded', options={'xatol': self.xatol}).x)
                points = [float(qref+7*k*dqref) for k in np.arange(-1,1+1)]
            else:
                dqref = 0.0
                points = [qref]
            Fval, Ferr = self.integrate_Fx(Fq, qref, points, self.qmin, self.qmax)
        else:
            Fval = np.zeros_like(s); Ferr = np.zeros_like(s)
            for i, si in enumerate(s):
                Fq = lambda q: self.VKC(theta,si,q)
                qref = self.minimize_Fx(Fq, self.qmin, self.qmax)
                if theta == 0:
                    dqref = abs(minimize_scalar(lambda dq: (self.beta*(Fq(qref + dq) - Fq(qref)) - 0.5)**2, bounds=(self.qmin - qref, self.qmax - qref), method='bounded', options={'xatol': self.xatol}).x)
                    points = [float(qref+7*k*dqref) for k in np.arange(-1,1+1)]
                else:
                    dqref = 0.0
                    points = [qref]
                Fval[i], Ferr[i] = self.integrate_Fx(Fq, qref, points, self.qmin, self.qmax)
        return Fval, Ferr

    # KC-RPMD free energy at theta along q
    def FKCthetaq(self, theta, q):
        if np.isscalar(q):
            Fs = lambda s: self.VKC(theta,s,q)
            sref = self.sdag #sref = min([self.s0, self.sdag, self.s1], key=Fs)
            if theta == 0:
                dsinv = max(np.sqrt(2*self.Aq(q))*self.ms*self.ws**2*np.abs(self.s1 - self.s0)/self.Kq(q), self.smax - self.smin)
                dsref = 1/dsinv
                points = [float(sref+7*k*dsref) for k in np.arange(-1,1+1)]
            else:
                dsref = 0.0
                points = [sref]
            Fval, Ferr = self.integrate_Fx(Fs, sref, points, self.smin, self.smax)
        else:
            Fval = np.zeros_like(q); Ferr = np.zeros_like(q)
            for i, qi in enumerate(q):
                Fs = lambda s: self.VKC(theta,s,qi)
                sref = self.sdag #sref = min([self.s0, self.sdag, self.s1], key=Fs)
                if theta == 0:
                    dsinv = max(np.sqrt(2*self.Aq(qi))*self.ms*self.ws**2*np.abs(self.s1 - self.s0)/self.Kq(qi), self.smax - self.smin)
                    dsref = 1/dsinv
                    points = [float(sref+7*k*dsref) for k in np.arange(-1,1+1)]
                else:
                    dsref = 0.0
                    points = [sref]
                Fval[i], Ferr[i] = self.integrate_Fx(Fs, sref, points, self.smin, self.smax)
        return Fval, Ferr

    # KC-RPMD free energy along s and q
    def FKCsq(self, s, q):
        V0 = self.VKC(-1,s,q)
        V1 = self.VKC(1,s,q)
        VKP = self.VKC(0,s,q)
        Fref = np.minimum.reduce([V0, V1, VKP])
        Fval = Fref - 1/self.beta*np.log(np.exp(-self.beta*(V0 - Fref))
                                         + np.exp(-self.beta*(V1 - Fref))
                                         + np.exp(-self.beta*(VKP - Fref)))
        Ferr = 0*Fval
        return Fval, Ferr

    # KC-RPMD free energy along y and s
    def FKCys(self, y, s):
        if np.isscalar(y) or np.isscalar(s):
            F0avg, F0err = self.FKCthetas(-1,s); F0avg = F0avg + self.Vr(-1,y)
            FKPavg, FKPerr = self.FKCthetas(0,s); FKPavg = FKPavg + self.Vr(0,y)
            F1avg, F1err = self.FKCthetas(1,s); F1avg = F1avg + self.Vr(1,y)
        else:
            F0avg, F0err = self.FKCthetas(-1,s); F0avg = F0avg[:,None] + self.Vr(-1,y)[None,:]; F0err = F0err[:,None]
            FKPavg, FKPerr = self.FKCthetas(0,s); FKPavg = FKPavg[:,None] + self.Vr(0,y)[None,:]; FKPerr = FKPerr[:,None]
            F1avg, F1err = self.FKCthetas(1,s); F1avg = F1avg[:,None] + self.Vr(1,y)[None,:]; F1err = F1err[:,None]
        Fref = np.minimum.reduce([F0avg, FKPavg, F1avg])
        Fval = Fref - 1/self.beta*np.log(np.exp(-self.beta*(F0avg - Fref))
                                         + np.exp(-self.beta*(FKPavg - Fref))
                                         + np.exp(-self.beta*(F1avg - Fref)))
        Ferr = (np.sqrt((np.exp(-self.beta*(F0avg - Fref))*F0err)**2
                        + (np.exp(-self.beta*(FKPavg - Fref))*FKPerr)**2
                        + (np.exp(-self.beta*(F1avg - Fref))*F1err)**2)
                /(np.exp(-self.beta*(F0avg - Fref))
                  + np.exp(-self.beta*(FKPavg - Fref))
                  + np.exp(-self.beta*(F1avg - Fref))))
        return Fval, Ferr

#    # KC-RPMD free energy along y and s
#    def FKCys(self, y, s):
#        if np.isscalar(y):
#            F0avg, F0err = self.FKCthetas(-1,s); F0avg = F0avg + self.Vr(-1,y)
#            FKPavg, FKPerr = self.FKCthetas(0,s); FKPavg = FKPavg + self.Vr(0,y)
#            F1avg, F1err = self.FKCthetas(1,s); F1avg = F1avg + self.Vr(1,y)
#        elif np.isscalar(s):
#            F0avg, F0err = self.FKCthetas(-1,s); F0avg = F0avg + self.Vr(-1,y); F0err = np.full_like(F0avg,F0err)
#            FKPavg, FKPerr = self.FKCthetas(0,s); FKPavg = FKPavg + self.Vr(0,y); FKPerr = np.full_like(FKPavg,FKPerr)
#            F1avg, F1err = self.FKCthetas(1,s); F1avg = F1avg + self.Vr(1,y); F1err = np.full_like(F1avg,F1err)
#        else:
#            F0avg, F0err = self.FKCthetas(-1,s); F0avg = F0avg[:,None] + self.Vr(-1,y)[None,:]; F0err = np.broadcast_to(F0err,F0avg.shape)
#            FKPavg, FKPerr = self.FKCthetas(0,s); FKPavg = FKPavg[:,None] + self.Vr(0,y)[None,:]; FKPerr = np.broadcast_to(FKPerr,FKPavg.shape)
#            F1avg, F1err = self.FKCthetas(1,s); F1avg = F1avg[:,None] + self.Vr(1,y)[None,:]; F1err = np.broadcast_to(F1err,F1avg.shape)
#        Fref = np.minimum.reduce([F0avg, FKPavg, F1avg])
#        Fval = Fref - 1/self.beta*np.log(np.exp(-self.beta*(F0avg - Fref))
#                                         + np.exp(-self.beta*(FKPavg - Fref))
#                                         + np.exp(-self.beta*(F1avg - Fref)))
#        Ferr = (np.sqrt((np.exp(-self.beta*(F0avg - Fref))*F0err)**2
#                        + (np.exp(-self.beta*(FKPavg - Fref))*FKPerr)**2
#                        + (np.exp(-self.beta*(F1avg - Fref))*F1err)**2)
#                /(np.exp(-self.beta*(F0avg - Fref))
#                  + np.exp(-self.beta*(FKPavg - Fref))
#                  + np.exp(-self.beta*(F1avg - Fref))))
#        return Fval, Ferr

    # KC-RPMD free energy along y and q
    def FKCyq(self, y, q):
        if np.isscalar(y) or np.isscalar(q):
            F0avg, F0err = self.FKCthetaq(-1,q); F0avg = F0avg + self.Vr(-1,y)
            FKPavg, FKPerr = self.FKCthetaq(0,q); FKPavg = FKPavg + self.Vr(0,y)
            F1avg, F1err = self.FKCthetaq(1,q); F1avg = F1avg + self.Vr(1,y)
        else:
            F0avg, F0err = self.FKCthetaq(-1,q); F0avg = F0avg[:,None] + self.Vr(-1,y)[None,:]; F0err = F0err[:,None]
            FKPavg, FKPerr = self.FKCthetaq(0,q); FKPavg = FKPavg[:,None] + self.Vr(0,y)[None,:]; FKPerr = FKPerr[:,None]
            F1avg, F1err = self.FKCthetaq(1,q); F1avg = F1avg[:,None] + self.Vr(1,y)[None,:]; F1err = F1err[:,None]
        Fref = np.minimum.reduce([F0avg, FKPavg, F1avg])
        Fval = Fref - 1/self.beta*np.log(np.exp(-self.beta*(F0avg - Fref))
                                         + np.exp(-self.beta*(FKPavg - Fref))
                                         + np.exp(-self.beta*(F1avg - Fref)))
        Ferr = (np.sqrt((np.exp(-self.beta*(F0avg - Fref))*F0err)**2
                        + (np.exp(-self.beta*(FKPavg - Fref))*FKPerr)**2
                        + (np.exp(-self.beta*(F1avg - Fref))*F1err)**2)
                /(np.exp(-self.beta*(F0avg - Fref))
                  + np.exp(-self.beta*(FKPavg - Fref))
                  + np.exp(-self.beta*(F1avg - Fref))))
        return Fval, Ferr

    # KC-RPMD free energy at theta
    def FKCtheta(self, theta):
        Fq = lambda q: self.FKCthetaq(theta,q)[0]
        if theta == -1:
            if not all(hasattr(self, name) for name in ["F0avg", "F0err"]):
                qref = self.minimize_Fx(Fq, self.qmin, self.qmax)
                points = [qref]
                Fval, Ferr = self.integrate_Fx(Fq, qref, points, self.qmin, self.qmax)
                self.F0avg, self.F0err = Fval, Ferr
            else:
                Fval, Ferr = self.F0avg, self.F0err
        elif theta == 0:
            if not all(hasattr(self, name) for name in ["FKPavg", "FKPerr"]):
                qref = self.minimize_Fx(Fq, self.qmin, self.qmax)
                points = [qref]
                Fval, Ferr = self.integrate_Fx(Fq, qref, points, self.qmin, self.qmax)
                self.FKPavg, self.FKPerr = Fval, Ferr
            else:
                Fval, Ferr = self.FKPavg, self.FKPerr
        elif theta == 1:
            if not all(hasattr(self, name) for name in ["F1avg", "F1err"]):
                qref = self.minimize_Fx(Fq, self.qmin, self.qmax)
                points = [qref]
                Fval, Ferr = self.integrate_Fx(Fq, qref, points, self.qmin, self.qmax)
                self.F1avg, self.F1err = Fval, Ferr
            else:
                Fval, Ferr = self.F1avg, self.F1err
        else:
            print('Invalid Theta!')
            return np.sqrt(-1)
        return Fval, Ferr

    # KC-RPMD free energy along s
    def FKCs(self, s):
        F0avg, F0err = self.FKCthetas(-1,s)
        FKPavg, FKPerr = self.FKCthetas(0,s)
        F1avg, F1err = self.FKCthetas(1,s)
        Fref = np.minimum.reduce([F0avg, FKPavg, F1avg])
        Fval = Fref - 1/self.beta*np.log(np.exp(-self.beta*(F0avg - Fref))
                                         + np.exp(-self.beta*(FKPavg - Fref))
                                         + np.exp(-self.beta*(F1avg - Fref)))
        Ferr = (np.sqrt((np.exp(-self.beta*(F0avg - Fref))*F0err)**2
                        + (np.exp(-self.beta*(FKPavg - Fref))*FKPerr)**2
                        + (np.exp(-self.beta*(F1avg - Fref))*F1err)**2)
                /(np.exp(-self.beta*(F0avg - Fref))
                  + np.exp(-self.beta*(FKPavg - Fref))
                  + np.exp(-self.beta*(F1avg - Fref))))
        return Fval, Ferr

    # KC-RPMD free energy along q
    def FKCq(self, q):
        F0avg, F0err = self.FKCthetas(-1,q)
        FKPavg, FKPerr = self.FKCthetas(0,q)
        F1avg, F1err = self.FKCthetas(1,q)
        Fref = np.minimum.reduce([F0avg, FKPavg, F1avg])
        Fval = Fref - 1/self.beta*np.log(np.exp(-self.beta*(F0avg - Fref))
                                         + np.exp(-self.beta*(FKPavg - Fref))
                                         + np.exp(-self.beta*(F1avg - Fref)))
        Ferr = (np.sqrt((np.exp(-self.beta*(F0avg - Fref))*F0err)**2
                        + (np.exp(-self.beta*(FKPavg - Fref))*FKPerr)**2
                        + (np.exp(-self.beta*(F1avg - Fref))*F1err)**2)
                /(np.exp(-self.beta*(F0avg - Fref))
                  + np.exp(-self.beta*(FKPavg - Fref))
                  + np.exp(-self.beta*(F1avg - Fref))))
        return Fval, Ferr

    # KC-RPMD free energy along y
    def FKCy(self, y):
        F0avg, F0err = self.FKCtheta(-1)
        FKPavg, FKPerr = self.FKCtheta(0)
        F1avg, F1err = self.FKCtheta(1)
        Fref = np.minimum.reduce([F0avg + self.Vr(-1,y), FKPavg + self.Vr(0,y), F1avg + self.Vr(1,y)])
        Fval = Fref - 1/self.beta*np.log(np.exp(-self.beta*(F0avg + self.Vr(-1,y) - Fref))
                                         + np.exp(-self.beta*(FKPavg + self.Vr(0,y) - Fref))
                                         + np.exp(-self.beta*(F1avg + self.Vr(1,y) - Fref)))
        Ferr = (np.sqrt((np.exp(-self.beta*(F0avg + self.Vr(-1,y) - Fref))*F0err)**2
                        + (np.exp(-self.beta*(FKPavg + self.Vr(0,y) - Fref))*FKPerr)**2
                        + (np.exp(-self.beta*(F1avg + self.Vr(1,y) - Fref))*F1err)**2)
                /(np.exp(-self.beta*(F0avg + self.Vr(-1,y) - Fref))
                  + np.exp(-self.beta*(FKPavg + self.Vr(0,y) - Fref))
                  + np.exp(-self.beta*(F1avg + self.Vr(1,y) - Fref))))
        return Fval, Ferr

    # KC-RPMD total free energy
    def FKC(self):
        if not all(hasattr(self, name) for name in ["FKCavg", "FKCerr"]):
            F0avg, F0err = self.FKCtheta(-1)
            FKPavg, FKPerr = self.FKCtheta(0)
            F1avg, F1err = self.FKCtheta(1)
            Fref = np.minimum.reduce([F0avg, FKPavg, F1avg])
            Fval = Fref - 1/self.beta*np.log(np.exp(-self.beta*(F0avg - Fref))
                                             + np.exp(-self.beta*(FKPavg - Fref))
                                             + np.exp(-self.beta*(F1avg - Fref)))
            Ferr = (np.sqrt((np.exp(-self.beta*(F0avg - Fref))*F0err)**2
                            + (np.exp(-self.beta*(FKPavg - Fref))*FKPerr)**2
                            + (np.exp(-self.beta*(F1avg - Fref))*F1err)**2)
                    /(np.exp(-self.beta*(F0avg - Fref))
                      + np.exp(-self.beta*(FKPavg - Fref))
                      + np.exp(-self.beta*(F1avg - Fref))))
            self.FKCavg, self.FKCerr = Fval, Ferr
        else:
             Fval, Ferr = self.FKCavg, self.FKCerr
        return Fval, Ferr


    ##############################
    ####### rate constants #######
    ##############################

    # Fermi golden rule rate constant
    def kGR(self):
        Fq = lambda q: self.Vq(q)
        qref = self.minimize_Fx(Fq, self.qmin, self.qmax)
        points=[qref]
        Fval, Ferr = self.integrate_Fx(Fq, qref, points, self.qmin, self.qmax)
        Pq = lambda q: np.exp(-self.beta*(Fq(q) - Fval))

        kGRq = lambda q: 2*np.pi*self.Kq(q)**2*np.sqrt(self.beta/(4*np.pi*self.lam))*np.exp(-self.beta*(self.lam + self.eps)**2/(4*self.lam))*Pq(q)
        kGRavg, kGRerr = quad(kGRq, self.qmin, self.qmax, epsabs=0.0, epsrel=self.epsrel, limit=self.limit, points=points)
        return kGRavg, kGRerr

    # Zusman rate constant
    def kZUS(self, tauL):
        Fq = lambda q: self.Vq(q)
        qref = self.minimize_Fx(Fq, self.qmin, self.qmax)
        points=[qref]
        Fval, Ferr = self.integrate_Fx(Fq, qref, points, self.qmin, self.qmax)
        Pq = lambda q: np.exp(-self.beta*(Fq(q) - Fval))

        kZUSq = lambda q: 2*np.pi*self.Kq(q)**2*np.sqrt(self.beta/(4*np.pi*self.lam))*np.exp(-self.beta*(self.lam + self.eps)**2/(4*self.lam))/(1 + 4*np.pi*self.Kq(q)**2*tauL/self.lam)*Pq(q)
        kZUSavg, kZUSerr = quad(kZUSq, self.qmin, self.qmax, epsabs=0.0, epsrel=self.epsrel, limit=self.limit, points=points)
        return kZUSavg, kZUSerr

    # Born-Oppenheimer TST rate constant along s
    def kBOs(self):
        Fs = lambda s: self.FBOs(s)[0]
        sref = min([self.s0, self.sdag], key=Fs)
        Fref = Fs(sref)
        exp_fun = lambda s: np.exp(-self.beta*(Fs(s) - Fref))
        Fref += - 1/self.beta*np.log(quad(exp_fun, self.smin, self.sdag, epsabs=0.0, epsrel=0.1, limit=self.limit)[0])
        Zval, Zerr = quad(exp_fun, self.smin, self.sdag, epsabs=self.beta*self.epsabs, epsrel=self.beta*abs(Fref)*self.epsrel, limit=self.limit)
        kBOavg = 1/np.sqrt(2*np.pi*self.beta*self.ms)*exp_fun(self.sdag)/Zval 
        kBOerr = kBOavg*(Zerr/Zval)
        return kBOavg, kBOerr

    # KC-RPMD TST rate constant along s
    def kKCs(self):
        Fs = lambda s: self.FKCs(s)[0]
        sref = min([self.s0, self.sdag], key=Fs)
        Fref = Fs(sref)
        exp_fun = lambda s: np.exp(-self.beta*(Fs(s) - Fref))
        Fref += - 1/self.beta*np.log(quad(exp_fun, self.smin, self.sdag, epsabs=0.0, epsrel=0.1, limit=self.limit)[0])
        Zval, Zerr = quad(exp_fun, self.smin, self.sdag, epsabs=self.beta*self.epsabs, epsrel=self.beta*abs(Fref)*self.epsrel, limit=self.limit)
        kKCavg = 1/np.sqrt(2*np.pi*self.beta*self.ms)*exp_fun(self.sdag)/Zval 
        kKCerr = kKCavg*(Zerr/Zval)
        return kKCavg, kKCerr

    # KC-RPMD TST rate constant along y
    def kKCy(self):
        Fy = lambda y: self.FKCy(y)[0]
        yref = min([-1.0, 0.0], key=Fy)
        Fref = Fy(yref)
        exp_fun = lambda y: np.exp(-self.beta*(Fy(y) - Fref))
        Fref += - 1/self.beta*np.log(quad(exp_fun, self.ymin, self.ydag, epsabs=0.0, epsrel=0.1, limit=self.limit)[0])
        Zval, Zerr = quad(exp_fun, self.ymin, self.ydag, epsabs=self.beta*self.epsabs, epsrel=self.beta*abs(Fref)*self.epsrel, limit=self.limit)
        kKCavg = 1/np.sqrt(2*np.pi*self.beta*self.my)*exp_fun(self.ydag)/Zval 
        kKCerr = kKCavg*(Zerr/Zval)
        return kKCavg, kKCerr

    ##########################################
    ####### probability distributions  #######
    ##########################################

    #Phw = np.exp(-beta * (kcrpmd_tst.Fg() - Fg)) # Hardwall potential probability from 0 to 1

    # Born-Oppenheimer probability distribution along s and q
    def PBOsq(self, s, q):
        Fnum = self.VBO(s,q)
        Fden = self.FBO()[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # Born-Oppenheimer probability distribution along s
    def PBOs(self, s):
        Fnum = self.FBOs(s)[0]
        Fden = self.FBO()[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # Born-Oppenheimer probability distribution along q
    def PBOq(self, q):
        Fnum = self.FBOq(q)[0]
        Fden = self.FBO()[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # Born-Oppenheimer probability distribution along q given s
    def PBOq_s(self, s, q):
        Fnum = self.VBO(s,q)
        if Fnum.ndim == 2:
            Fden = self.FBOs(s)[0][None,:]
        else:
            Fden = self.FBOs(s)[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # Born-Oppenheimer probability distribution along s given q
    def PBOs_q(self, s, q):
        Fnum = self.VBO(s,q)
        if Fnum.ndim == 2:
            Fden = self.FBOq(q)[0][:,None]
        else:
            Fden = self.FBOq(q)[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # KC-RPMD probability distribution along y and s
    def PKCys(self, y, s):
        Fnum = self.FKCys(y,s)[0]
        Fden = self.FKC()[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # KC-RPMD probability distribution along y and s
    def PKCyq(self, y, q):
        Fnum = self.FKCyq(y,q)[0]
        Fden = self.FKC()[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # KC-RPMD probability distribution along s and q
    def PKCsq(self, s, q):
        Fnum = self.FKCsq(s,q)[0]
        Fden = self.FKC()[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # KC-RPMD probability distribution along y
    def PKCy(self, y):
        Fnum = self.FKCy(y)[0]
        Fden = self.FKC()[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # KC-RPMD probability distribution along s
    def PKCs(self, s):
        Fnum = self.FKCs(s)[0]
        Fden = self.FKC()[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # KC-RPMD probability distribution along q
    def PKCq(self, q):
        Fnum = self.FKCq(q)[0]
        Fden = self.FKC()[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # KC-RPMD probability distribution along s given y
    def PKCs_y(self, y, s):
        Fnum = self.FKCys(y,s)[0]
        if Fnum.ndim == 2:
            Fden = self.FKCy(y)[0][None,:]
        else:
            Fden = self.FKCy(y)[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # KC-RPMD probability distribution along y given s
    def PKCy_s(self, y, s):
        Fnum = self.FKCys(y,s)[0]
        if Fnum.ndim == 2:
            Fden = self.FKCs(s)[0][:,None]
        else:
            Fden = self.FKCs(s)[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # KC-RPMD probability distribution along q given y
    def PKCq_y(self, y, q):
        Fnum = self.FKCyq(y,q)[0]
        if Fnum.ndim == 2:
            Fden = self.FKCy(y)[0][None,:]
        else:
            Fden = self.FKCy(y)[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # KC-RPMD probability distribution along y given q
    def PKCy_q(self, y, q):
        Fnum = self.FKCyq(y,q)[0]
        if Fnum.ndim == 2:
            Fden = self.FKCq(q)[0][:,None]
        else:
            Fden = self.FKCq(q)[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # KC-RPMD probability distribution along q given s
    def PKCq_s(self, s, q):
        Fnum = self.FKCsq(s,q)[0]
        if Fnum.ndim == 2:
            Fden = self.FKCs(s)[0][None,:]
        else:
            Fden = self.FKCs(s)[0]
        return np.exp(-self.beta*(Fnum - Fden))

    # KC-RPMD probability distribution along s given q
    def PKCs_q(self, s, q):
        Fnum = self.FKCsq(s,q)[0]
        if Fnum.ndim == 2:
            Fden = self.FKCq(q)[0][:,None]
        else:
            Fden = self.FKCq(q)[0]
        return np.exp(-self.beta*(Fnum - Fden))

    #def sample_BO():

####### DON'T FORGET THIS ONE, PRETTY USEFUL TRICK #######
    # KC-RPMD kinked pair potential Vkp (without state independent part V)
#    def Vkp(self, s, q):
#        #print('yes2')
#        self.betaKtol = 1.0e-2
#        self.betaDtol = 1.0e-5
#        if np.isscalar(s) and np.isscalar(q):
#            is_scalar = True
#            s = np.atleast_1d(s)
#        else:
#            is_scalar = False
#
#        K = self.K(s,q); bK = self.beta*K
#        D = self.D(s,q); bD = self.beta*D
#        Vz = self.Vz(s,q)
#
#        A1 = np.zeros_like(K)
#        A2 = np.zeros_like(K)
#        A3 = np.zeros_like(K)
#
#        Vkp = np.zeros_like(K)
#        mask_ad = (bK > self.betaKtol)
#        mask_nad = (bK <= self.betaKtol) & (np.abs(bD) > self.betaDtol)
#        mask_hole = (bK <= self.betaKtol) & (np.abs(bD) <= self.betaDtol)
#
#        Vkp[mask_ad] = -Vz[mask_ad] - np.log(1
#                                             + np.exp(-2*self.beta*Vz[mask_ad])
#                                             - np.exp(-self.beta*(Vz + D)[mask_ad])
#                                             - np.exp(-self.beta*(Vz - D)[mask_ad]))/self.beta
#
#        A1[mask_nad] = np.sinh(bD[mask_nad])/(bD[mask_nad])
#        A1[mask_hole] = (1 + bD[mask_hole]**2/6 + bD[mask_hole]**4/120 + bD[mask_hole]**6/5040)
#
#        A2[mask_nad] = (3*np.cosh(bD[mask_nad])/(bD[mask_nad]**2)
#                        - 3*np.sinh(bD[mask_nad])/(bD[mask_nad]**3))
#        A2[mask_hole] = (1 + bD[mask_hole]**2/10 + bD[mask_hole]**4/280 + bD[mask_hole]**6/15120)
#
#        A3[mask_nad] = (15*np.sinh(bD[mask_nad])/(bD[mask_nad]**3)
#                        - 45*np.cosh(bD[mask_nad])/(bD[mask_nad]**4)
#                        + 45*np.sinh(bD[mask_nad])/(bD[mask_nad]**5))
#        A3[mask_hole] = (1 + bD[mask_hole]**2/14 + bD[mask_hole]**4/504 + bD[mask_hole]**6/30240)
#
#        Vkp[~mask_ad] = -np.log(bK[~mask_ad]**2*A1[~mask_ad]
#                                + bK[~mask_ad]**4*A2[~mask_ad]/12
#                                + bK[~mask_ad]**6*A2[~mask_ad]/360)/self.beta
#
#        return Vkp.item() if is_scalar else Vkp





    ##########################################
    ####### NVT configuration sampling #######
    ##########################################




#    def tst_s(self):
#        s_ar = self.s_array(); s_ar = s_ar[:1+np.argwhere(s_ar == self.sdag)[0,0]]
#        exp_arg = -self.beta * self.Fs(s_ar)
#        exp_shift = np.max(exp_arg) - 500.
#        return 1 / np.sqrt(2 * np.pi * self.beta * self.ms) * np.exp(exp_arg[-1] - exp_shift) / np.trapezoid(np.exp(exp_arg - exp_shift), s_ar)




    ############################################################
    ####### KC-RPMD potentials, free energies, and rates #######
    ############################################################


#    def FKCtheta(self, theta):
#        Fq = lambda q: self.FKCthetaq(theta,q)[0]
#        qref = minimize_scalar(lambda q: self.beta*Fq(q), bounds=(self.qmin, self.qmax), method='bounded', options={'xatol': 1e-9}).x
#        points = [qref]
#        Fval, Ferr = self.integrate_q(Fq, qref, points)
#        return Fval, Ferr



#    def F(self):
#        if not all(hasattr(self, name) for name in ["F0avg", "F0err", "FKPavg", "FKPerr", "F1avg", "F1err"]):
#            self.compute_Ftheta()
#        Fy = lambda y: self.FKCy(y)[0]
#        yref = self.ydag
#        points = [-1.5,-0.5,0.5,1.5]
#
#        Fref = Fy(yref)
#        exp_fun = lambda y: np.exp(-self.beta*(Fy(y) - Fref))
#        Fref += - 1/self.beta*np.log(quad(exp_fun, -1.6, 1.6, epsabs=0.0, epsrel=0.1, limit=self.limit, points=points)[0])
#        val, err = quad(exp_fun, -1.6, 1.6, epsabs=self.beta*self.epsabs, epsrel=self.beta*abs(Fref)*self.epsrel, limit=self.limit, points=points)
#        Fval = Fref - 1/self.beta*np.log(val)
#        Ferr = abs(err/(self.beta*val))
#        return Fval, Ferr


#        exp_fun = 
#        val, err = quad(exp_fun, self.smin, self.smax, epsabs=self.beta*self.epsabs, epsrel=self.beta*abs(Fref)*self.epsrel, limit=self.limit, points=points)
#
#        Fval, Ferr = self.integrate_s(Fs, sref, points)
#
#        return Fval, Ferr
#
#        y_ar = self.y_array()
#        exp_arg = -self.beta * self.Fy(y_ar)
#        exp_shift = np.max(exp_arg) - 500.
#        return -np.log(np.trapezoid(np.exp(exp_arg - exp_shift), y_ar)) / self.beta - exp_shift / self.beta



        #self.dV0s = lambda s, sref=0.: 0.5*self.ms*self.ws**2*(s - sref)*(s + sref - 2*self.s0)
        #self.dV1s = lambda s, sref=0.: 0.5*self.ms*self.ws**2*(s - sref)*(s + sref - 2*self.s1)
        #self.dVs = lambda s, sref=0.: 0.5*self.ms*self.ws**2*(s - sref)*(s + sref - self.s0 - self.s1)
        #self.dlnKq = lambda q, qref=0.: self.lnKq(q) - self.lnKq(qref)
        #self.dlnSq = lambda q, qref=0.: self.lnSq(q) - self.lnSq(qref)
        #self.dlnSq = lambda q, qref=0.: np.piecewise(q,
        #                                    [self.beta*np.exp(self.lnKq(q)) >= 1. and self.beta*np.exp(self.lnKq(qref)) >= 1.,
        #                                    self.beta*np.exp(self.lnKq(q)) >= 1. and self.beta*np.exp(self.lnKq(qref)) < 1.],
        #                                    [lambda q: (-2*self.c*self.beta*np.exp(self.lnKq(qref))( - 1)
        #                                                - np.log(1 + np.exp(-2*self.c*(self.beta*np.exp(self.lnKq(q)) - 1)))),
        #                                     lambda q: -np.log(1 + np.exp(2*self.c*(self.beta*np.exp(self.lnKq(q)) - 1)))])


        #self.Aq = lambda q: np.piecewise(q,
        #                                 [self.beta*self.Kq(q) >= 1.,
        #                                  self.beta*self.Kq(q) < 1.],
        #                                 [lambda q: self.a*(np.exp(-2*self.c*(self.beta*self.Kq(q) - 1))
        #                                                    /(1 + np.exp(-2*self.c*(self.beta*self.Kq(q) - 1)))),
        #                                  lambda q: self.a/(1 + np.exp(2*self.c*(self.beta*self.Kq(q) - 1)))])
        #self.Cq = lambda q: np.piecewise(q,
        #                                 [self.beta*self.Kq(q) >= 1.,
        #                                  self.beta*self.Kq(q) < 1.],
        #                                 [lambda q: 1 + (self.Aq(q)/np.pi*self.eta - 1)*(np.exp(-2*self.c*(self.beta*self.Kq(q) - 1))
        #                                                                                 /(1 + np.exp(-2*self.c*(self.beta*self.Kq(q) - 1)))),
        #                                  lambda q: 1 + (self.Aq(q)/np.pi*self.eta - 1)/(1 + np.exp(2*self.c*(self.beta*self.Kq(q) - 1)))])
    
        #self.Cq = lambda q: np.piecewise(q,[self.beta*self.Kq(q) >= 1.,self.beta*self.Kq(q) < 1.],
        #                                 [lambda q: (1 + (self.eta*np.sqrt(self.a/np.pi)*np.exp(-self.c*(self.beta*self.Kq(q) - 1))
        #                                                  /np.sqrt(1 + np.exp(-2*self.c*(self.beta*self.Kq(q) - 1))) - 1)
        #                                                  *np.exp(-2*self.c*(self.beta*self.Kq(q) - 1))
        #                                                  /(1 + np.exp(-2*self.c*(self.beta*self.Kq(q) - 1)))),
        #                                  lambda q:(1 + (self.eta*np.sqrt(self.a/np.pi)
        #                                                 /np.sqrt(1 + np.exp(2*self.c*(self.beta*self.Kq(q) - 1))) - 1)
        #                                                 /(1 + np.exp(2*self.c*(self.beta*self.Kq(q) - 1))))])
        #
        ##    def dV0(self, s, q, sref=0., qref=0.):
#        if np.isscalar(s) or np.isscalar(q):
#            return self.dV0s(s,sref) + self.dVq(q,qref)
#        else:
#            S,Q = np.meshgrid(s,q)
#            return self.dV0s(S,sref) + self.dVq(Q,qref)
#
#    def dV1(self, s, q, sref=0., qref=0.):
#        if np.isscalar(s) or np.isscalar(q):
#            return self.dV1s(s,sref) + self.dVq(q,qref)
#        else:
#            S,Q = np.meshgrid(s,q)
#            return self.dV1s(S,sref) + self.dVq(Q,qref)
#
#    def dV(self, s, q, sref=0., qref=0.):
#        if np.isscalar(s) or np.isscalar(q):
#            return self.dVs(s,sref) + self.dVq(q,qref)
#        else:
#            S,Q = np.meshgrid(s,q)
#            return self.dVs(S,sref) + self.dVq(Q,qref)

#    def VKP(self, s, q):
#        if np.isscalar(s) and np.isscalar(q):
#            is_scalar = True
#            s = np.atleast_1d(s)
#        else:
#            is_scalar = False
#
#        V = self.V(s,q)
#        D = self.D(s,q); bD = self.beta*D
#        K = self.K(s,q); bK = self.beta*K
#        Vz = self.Vz(s,q)
#
#        A1 = np.zeros_like(V)
#        A2 = np.zeros_like(V)
#        A3 = np.zeros_like(V)
#
#        VKP = np.zeros_like(V)
#        mask_ad = (bK > self.betaKtol)
#        mask_nad = (bK <= self.betaKtol) & (np.abs(bD) > self.betaDtol)
#        mask_hole = (bK <= self.betaKtol) & (np.abs(bD) <= self.betaDtol)
#
#        VKP[mask_ad] = (V - Vz)[mask_ad] - np.log(1
#                                            + np.exp(-2*self.beta*Vz[mask_ad])
#                                            - np.exp(-self.beta*(Vz + D)[mask_ad])
#                                            - np.exp(-self.beta*(Vz - D)[mask_ad]))/self.beta
#
#        A1[mask_nad] = np.sinh(bD[mask_nad])/(bD[mask_nad])
#        A1[mask_hole] = (1 + bD[mask_hole]**2/6 + bD[mask_hole]**4/120 + bD[mask_hole]**6/5040)
#
#        A2[mask_nad] = (3*np.cosh(bD[mask_nad])/(bD[mask_nad]**2)
#                        - 3*np.sinh(bD[mask_nad])/(bD[mask_nad]**3))
#        A2[mask_hole] = (1 + bD[mask_hole]**2/10 + bD[mask_hole]**4/280 + bD[mask_hole]**6/15120)
#
#        A3[mask_nad] = (15*np.sinh(bD[mask_nad])/(bD[mask_nad]**3)
#                        - 45*np.cosh(bD[mask_nad])/(bD[mask_nad]**4)
#                        + 45*np.sinh(bD[mask_nad])/(bD[mask_nad]**5))
#        A3[mask_hole] = (1 + bD[mask_hole]**2/14 + bD[mask_hole]**4/504 + bD[mask_hole]**6/30240)
#
#        #print(f'arg = {np.log(bK[~mask_ad]**2*A1[~mask_ad] + bK[~mask_ad]**4*A2[~mask_ad]/12 + bK[~mask_ad]**6*A2[~mask_ad]/360)}')
#        VKP[~mask_ad] = V[~mask_ad] - np.log(bK[~mask_ad]**2*A1[~mask_ad]
#                                             + bK[~mask_ad]**4*A2[~mask_ad]/12
#                                             + bK[~mask_ad]**6*A2[~mask_ad]/360)/self.beta
#
#        return VKP.item() if is_scalar else VKP

#    def w(self, s, q):
#        return 2*self.D(s,q)/self.K(s,q)

    #def A(self, s, q):
    #    if np.isscalar(s) and np.isscalar(q):
    #        is_scalar = True
    #        s = np.atleast_1d(s)
    #    else:
    #        is_scalar = False
    #    K = self.K(s,q); bK = self.beta*K
    #    A = np.zeros_like(K)
    #    mask_ad = (bK > 1.0)
    #    mask_nad = (bK <= 1.0)
    #    A[mask_ad] = (self.a*np.exp(-2*self.c*(bK[mask_ad] - 1.0))
    #                  /(1 + np.exp(-2*self.c*(bK[mask_ad] - 1.0))))
    #    A[mask_nad] = self.a/(1 + np.exp(2*self.c*(bK[mask_nad] - 1.0)));
    #    return A.item() if is_scalar else A

#    def C(self, s, q):
#        if np.isscalar(s) and np.isscalar(q):
#            is_scalar = True
#            s = np.atleast_1d(s)
#        else:
#            is_scalar = False
#        K = self.K(s,q); bK = self.beta*K
#        C = np.zeros_like(K)
#        mask_ad = (bK > 1.0)
#        mask_nad = (bK <= 1.0)
#        C[mask_ad] = (1 + (self.eta*np.sqrt(self.a/np.pi)*np.exp(-self.c*(bK[mask_ad] - 1.0))
#                           /np.sqrt(1 + np.exp(-2*self.c*(bK[mask_ad] - 1.0))) - 1)
#                          *np.exp(-2*self.c*(bK[mask_ad] - 1.0))
#                          /(1 + np.exp(-2*self.c*(bK[mask_ad] - 1.0))))
#        C[mask_nad] = (1 + (self.eta*np.sqrt(self.a/np.pi)
#                            /np.sqrt(1 + np.exp(2*self.c*(bK[mask_nad] - 1.0))) - 1)
#                           /(1 + np.exp(2*self.c*(bK[mask_nad] - 1.0))))
#        return C.item() if is_scalar else C

#    # Born-Oppenheimer total free energy
#    def FBO(self):
#        Fs = lambda s: self.FBOs(s)[0]
#        sref = min([self.s0, self.sdag, self.s1], key=Fs)
#        points = [sref]
#        Fval, Ferr = self.integrate_s(Fs, sref, points)
#        return Fval, Ferr

#    def integrate_s(self, Fs, sref, points, smin, smax):
#        Fref = Fs(sref)
#        exp_fun = lambda s: np.exp(-self.beta*(Fs(s) - Fref))
#        Fref += - 1/self.beta*np.log(quad(exp_fun, smin, smax, epsabs=0.0, epsrel=0.1, limit=self.limit, points=points)[0])
#        val, err = quad(exp_fun, smin, smax, epsabs=self.beta*self.epsabs, epsrel=self.beta*abs(Fref)*self.epsrel, limit=self.limit, points=points)
#        Fval = Fref - 1/self.beta*np.log(val)
#        Ferr = abs(err/(self.beta*val))
#        return Fval, Ferr
#
#    def integrate_q(self, Fq, qref, points):
#        Fref = Fq(qref)
#        exp_fun = lambda q: np.exp(-self.beta*(Fq(q) - Fref))
#        Fref += - 1/self.beta*np.log(quad(exp_fun, self.qmin, self.qmax, epsabs=0.0, epsrel=0.1, limit=self.limit, points=points)[0])
#        val, err = quad(exp_fun, self.qmin, self.qmax, epsabs=self.beta*self.epsabs, epsrel=self.beta*abs(Fref)*self.epsrel, limit=self.limit, points=points)
#        Fval = Fref - 1/self.beta*np.log(val)
#        Ferr = abs(err/(self.beta*val))
#        return Fval, Ferr

####### DON'T FORGET THIS ONE, PRETTY USEFUL TRICK #######

#    # KC-RPMD kinked pair potential Vkp (without state independent part V)
    def Vkp2(self, s, q):
        self.betaKtol = 1.0e-2
        self.betaDtol = 1.0e-5
        if np.isscalar(s) and np.isscalar(q):
            is_scalar = True
            s = np.atleast_1d(s)
        else:
            is_scalar = False

        K = self.K(s,q); bK = self.beta*K
        D = self.D(s,q); bD = self.beta*D
        Vz = self.Vz(s,q)

        A1 = np.zeros_like(K)
        A2 = np.zeros_like(K)
        A3 = np.zeros_like(K)

        Vkp = np.zeros_like(K)
        mask_ad = (bK > self.betaKtol)
        mask_nad = (bK <= self.betaKtol) & (np.abs(bD) > self.betaDtol)
        mask_hole = (bK <= self.betaKtol) & (np.abs(bD) <= self.betaDtol)

        Vkp[mask_ad] = -Vz[mask_ad] - np.log(1
                                             + np.exp(-2*self.beta*Vz[mask_ad])
                                             - np.exp(-self.beta*(Vz + D)[mask_ad])
                                             - np.exp(-self.beta*(Vz - D)[mask_ad]))/self.beta

        A1[mask_nad] = np.sinh(bD[mask_nad])/(bD[mask_nad])
        A1[mask_hole] = (1 + bD[mask_hole]**2/6 + bD[mask_hole]**4/120 + bD[mask_hole]**6/5040)

        A2[mask_nad] = (3*np.cosh(bD[mask_nad])/(bD[mask_nad]**2)
                        - 3*np.sinh(bD[mask_nad])/(bD[mask_nad]**3))
        A2[mask_hole] = (1 + bD[mask_hole]**2/10 + bD[mask_hole]**4/280 + bD[mask_hole]**6/15120)

        A3[mask_nad] = (15*np.sinh(bD[mask_nad])/(bD[mask_nad]**3)
                        - 45*np.cosh(bD[mask_nad])/(bD[mask_nad]**4)
                        + 45*np.sinh(bD[mask_nad])/(bD[mask_nad]**5))
        A3[mask_hole] = (1 + bD[mask_hole]**2/14 + bD[mask_hole]**4/504 + bD[mask_hole]**6/30240)

        Vkp[~mask_ad] = -np.log(bK[~mask_ad]**2*A1[~mask_ad]
                                + bK[~mask_ad]**4*A2[~mask_ad]/12
                                + bK[~mask_ad]**6*A2[~mask_ad]/360)/self.beta

        return Vkp.item() if is_scalar else Vkp

    def Vkp3(self, s, q):
        V0 = self.V0(s, q)
        V1 = self.V1(s, q)
        K = self.K(s, q)
        Vz = self.Vz(s, q)
        V = self.V(s, q)

        VKP = np.zeros_like(K)
        mask_ad = (self.beta * K > 1e-3)
        mask_nad = (self.beta * K <= 1e-3) & (self.beta * np.abs(V0 - V1) > 1e-7)
        mask_hole = (self.beta * K <= 1e-3) & (self.beta * np.abs(V0 - V1) <= 1e-7)
        #mask_ad = (self.beta * K > 1e-2)
        #mask_nad = (self.beta * K <= 1e-2) & (self.beta * np.abs(V0 - V1) > 1e-5)
        #mask_hole = (self.beta * K <= 1e-2) & (self.beta * np.abs(V0 - V1) <= 1e-5)

        VKP[mask_ad] = -Vz[mask_ad] - np.log(1
                                            + np.exp(-self.beta * (V + Vz - V + Vz)[mask_ad])
                                            - np.exp(-self.beta * (V0 - V + Vz)[mask_ad])
                                            - np.exp(-self.beta * (V1 - V + Vz)[mask_ad])) / self.beta

        VKP[mask_nad] =  - np.log((self.beta * K[mask_nad])**2
                                                                     * np.sinh(0.5 * self.beta
                                                                               * (V0[mask_nad] - V1[mask_nad]))
                                                                     / (0.5 * self.beta
                                                                        * (V0[mask_nad] - V1[mask_nad]))) / self.beta

        VKP[mask_hole] =  - np.log((self.beta * K[mask_hole])**2) / self.beta

        return VKP
#
#    # KC-RPMD kinked pair potential Vkp (without state independent part V)
#    def Vkp0(self, s, q):
#        if np.isscalar(s) and np.isscalar(q):
#            is_scalar = True
#            s = np.atleast_1d(s)
#        else:
#            is_scalar = False
#
#        K = self.K(s,q); bK = self.beta*K
#        D = self.D(s,q); bD = self.beta*D
#        Vz = self.Vz(s,q)
#
#        A1 = np.zeros_like(K)
#        A2 = np.zeros_like(K)
#        A3 = np.zeros_like(K)
#
#        Vkp = np.zeros_like(K)
#        mask_ad = (bK > self.betaKtol)
#        mask_nad = (bK <= self.betaKtol) & (np.abs(bD) > self.betaDtol)
#        mask_hole = (bK <= self.betaKtol) & (np.abs(bD) <= self.betaDtol)
#
#        Vkp[mask_ad] = -Vz[mask_ad] - np.log(1
#                                             + np.exp(-2*self.beta*Vz[mask_ad])
#                                             - np.exp(-self.beta*(Vz + D)[mask_ad])
#                                             - np.exp(-self.beta*(Vz - D)[mask_ad]))/self.beta
#
#        A1[mask_nad] = np.sinh(bD[mask_nad])/(bD[mask_nad])
#        A1[mask_hole] = (1 + bD[mask_hole]**2/6 + bD[mask_hole]**4/120 + bD[mask_hole]**6/5040)
#
#        A2[mask_nad] = (3*np.cosh(bD[mask_nad])/(bD[mask_nad]**2)
#                        - 3*np.sinh(bD[mask_nad])/(bD[mask_nad]**3))
#        A2[mask_hole] = (1 + bD[mask_hole]**2/10 + bD[mask_hole]**4/280 + bD[mask_hole]**6/15120)
#
#        A3[mask_nad] = (15*np.sinh(bD[mask_nad])/(bD[mask_nad]**3)
#                        - 45*np.cosh(bD[mask_nad])/(bD[mask_nad]**4)
#                        + 45*np.sinh(bD[mask_nad])/(bD[mask_nad]**5))
#        A3[mask_hole] = (1 + bD[mask_hole]**2/14 + bD[mask_hole]**4/504 + bD[mask_hole]**6/30240)
#
#        Vkp[~mask_ad] = -np.log(bK[~mask_ad]**2*A1[~mask_ad]
#                                + bK[~mask_ad]**4*A2[~mask_ad]/12
#                                + bK[~mask_ad]**6*A2[~mask_ad]/360)/self.beta
#
#        return Vkp.item() if is_scalar else Vkp
#
