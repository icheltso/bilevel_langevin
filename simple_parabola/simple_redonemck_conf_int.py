# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:17:03 2023

Solve bilevel problem as system of SDEs:
    min_lam  C(x(lam),lam)
    s.t. x(lam) = argmin_x F(x,lam)
2-D TEST PROBLEM:
    C(x,lam) = (1/2) * (x-2)^2
    F(x,lam) = (1/2) * (x-1)^2 + (1/2) * lam^2 * x^2
    
    optimal lambda = 0

@author: ichel
"""

import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import random
import colorsys


class Model:
    """Stochastic model constants."""
    SIGMA_X = 0.1
    SIGMA_lam = 0.1
    """Epsilon serves to differentiate between the two timesteps of the inner and outer problem"""
    EPSILON = 0.05


def mu_x(x: float, lam: float, _t: float) -> float:
    """Implement the drift term for x (inner, quick sde)."""
    mux = -((x-1) + (lam**2) * x)
    return mux

def mu_lam(x: float, lam: float, _t: float) -> float:
    """Implement the drift term for lambda (outer, slow sde)."""
    mul = 2*lam*x*(1/(1+lam**2))*(x-2)
    return mul


def sigma_x(x: float, lam: float, _t: float) -> float:
    """Implement the diffusion coefficient for x (inner problem)."""
    return Model.SIGMA_X

def sigma_lam(x: float, lam: float, _t: float) -> float:
    """Implement the diffusion coefficient for lambda (outer problem)."""
    return Model.SIGMA_lam


def dW(delta_t: float) -> float:
    """Sample a random number at each call."""
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

def nablam_F(x: float, lam: float, _t: float) -> float:
    """Return the gradient of F w.r.t. lambda"""
    return lam * x**2

def mckean_term(xvs, lam: float, t: float, DT: float, num_xs: int) -> float:
    """Return the extra term from the Mckean-Vlasov SDE"""
    
    xvals = np.zeros(num_xs)
    nablamFs = np.zeros(num_xs)
    pdfx = np.zeros(num_xs)
    integ = 0
    eps0 = Model.EPSILON
    
    for i in range(num_xs):
        xvals[i] = xvs[i] + mu_x(xvs[i],lam,t)*DT/eps0 + sigma_x(xvs[i], lam, t) * dW(DT)
        #print("X_{n+1} = " + str(xnew))
        nablamFs[i] = nablam_F(xvals[i],lam,t)
        #("nabla_lam_F" + str(nablamF))
        #pdfx[i] = distrib.pdf(xvals[i])
        #print('prob = ' + str(probdx))
        integ += nablamFs[i]
        
    integ = integ/num_xs

    #integ = scipy.integrate.simpson(nablamFs*pdfx,xvals)
    #integ = scipy.integrate.trapezoid(nablamFs*pdfx,xvals)
    print(nablamFs)
    print(xvals)
    print(integ)
    
    return integ, xvals


def run_sde():
    """ Return the result of one full simulation."""
    T_INIT = 0
    T_END = 2
    eps = Model.EPSILON
    
    N = 1000  # Compute at 1000 grid points
    DT = float(T_END - T_INIT) / N
    TS = np.arange(T_INIT, T_END + DT, DT)


    X_INIT = 0
    L_INIT = 1

    xs = np.zeros(TS.size)
    xs[0] = X_INIT
    ls = np.zeros(TS.size)
    ls[0] = L_INIT
    for i in range(1, TS.size):
        t = T_INIT + (i - 1) * DT
        x = xs[i - 1]
        lam = ls[i - 1]
        xs[i] = x + mu_x(x, lam, t) * DT/eps + sigma_x(x, lam, t) * dW(DT)
        ls[i] = lam + mu_lam(x, lam, t) * DT + sigma_lam(x, lam, t) * dW(DT)
        
        

    return TS, xs, ls

def run_mckean(num_xs: int):
    """ Return the result of one full simulation."""
    T_INIT = 0
    T_END = 2
    eps = Model.EPSILON
    
    N = 1000  # Compute at 1000 grid points
    DT = float(T_END - T_INIT) / N
    TS = np.arange(T_INIT, T_END + DT, DT)


    X_INIT = 0
    L_INIT = 1
    
    cl_sig = 1
    X_CLOUD = np.random.normal(0,cl_sig,num_xs)

    """Initialize arrays to save intermediate results"""
    #xs = np.zeros((num_xs,TS.size))
    xs = np.zeros(TS.size)
    #xs[:,0] = X_INIT
    xs[0] = X_INIT
    ls = np.zeros(TS.size)
    ls[0] = L_INIT
    """Initialize first batch of x-values"""
    #xvs = X_INIT * np.ones(num_xs)
    xvs = X_CLOUD
    """index of X that is used to update lambda."""
    #x_arg = random.choice(range(num_xs))
    x_arg = 0
    for i in range(1, TS.size):
        t = T_INIT + (i - 1) * DT
        lam = ls[i - 1]
        """Generate multiple X values for next timestep alongside the integral term"""
        integ, xvs = mckean_term(xvs, lam, t, DT, num_xs)
        #xs[:,i] = xvs
        x = xvs[x_arg]
        xs[i] = x
        """Encode mckean term"""
        mvt = (-1/eps) * nablam_F(x,lam,t) + (1/eps)*integ
        ls[i] = lam + mvt * DT + mu_lam(x, lam, t) * DT + sigma_lam(x, lam, t) * dW(DT)
        
        

    return TS, xs, ls

"""Generate colour palette with n colours (For mckean plots)"""
def get_N_HexCol(N):
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

def get_CI(xs,ls):
    """Generate 95% confidence interval for graphs"""
    pt_ct = np.size(xs,0)
    xsm = np.mean(xs, axis = 0)
    lsm = np.mean(ls, axis = 0)
    xstd = np.std(xs, axis = 0)
    lstd = np.std(xs, axis = 0)
    
    print(pt_ct)
    
    XCI = 1.96 * xstd / np.sqrt(pt_ct)
    print(XCI)
    #XCI = xstd
    XCIL = xsm - XCI
    XCIU = xsm + XCI
    
    LCI = 1.96 * lstd / np.sqrt(pt_ct)
    #LCI = lstd
    LCIL = lsm - LCI
    LCIU = lsm + LCI
    
    
    return xsm, XCIL, XCIU, lsm, LCIL, LCIU

def plot_simulations_OLD(num_sims: int):
    fig = plt.figure()
    """ Plot several simulations in one image."""
    for i in range(num_sims):
        TS, xs, ls = run_sde()
        if i == num_sims - 1:
            plt.plot(TS, xs, label='x')
            plt.plot(TS, ls, label='$\lambda$')
        else:
            plt.plot(TS, xs)
            plt.plot(TS, ls)
        

    plt.xlabel("time")
    plt.ylabel("x,$\lambda$")
    plt.legend()
    plt.title("Simulations of Stocbio SDE")
    plt.show()
    return TS, xs, ls

def plot_simulations(num_sims: int):
    """Plot results of stocbio run alongside confidence intervals"""
    palette_num = get_N_HexCol(num_sims)
    fig, ax = plt.subplots(1)
    fig_ci, ax_ci = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    fig2_ci, ax2_ci = plt.subplots(1)
    #fig_t, ax_t = plt.subplots(1)
    """ Plot several simulations in one image."""
    xs_many = []
    ls_many = []
    for i in range(num_sims):
        TS, xs, ls = run_sde()
        xs_many.append(xs)
        ls_many.append(ls)
        ax.plot(TS, ls, palette_num[i])
        #ax2.plot(TS, np.mean(xs, axis = 0), palette_num[i])
        ax2.plot(TS, xs, palette_num[i])
        #for j in range(num_xs):
        #    ax2.plot(TS, xs[j], palette_num[i])
        
    xsnp = np.asarray(xs_many)
    lsnp = np.asarray(ls_many)
    xsm, XCIL, XCIU, lsm, LCIL, LCIU = get_CI(xsnp,lsnp)
    ax_ci.plot(TS,lsm)
    ax_ci.fill_between(TS, LCIL, LCIU, color="blue", alpha=.15)
    ax2_ci.plot(TS,xsm)
    ax2_ci.fill_between(TS, XCIL, XCIU, color="blue", alpha=.15)
    
    #ax_t.hist(xsnp[:,2000])
    #fig_t.show()
    #print(np.std(xsnp[:,2000]))
    
    ax.set_xlabel("time")
    ax.set_ylabel("$\lambda$")
    ax.legend()
    ax.set_title("Simulations of Stocbio SDE - $\lambda$-values")
    fig.show()
    
    ax_ci.set_xlabel("time")
    ax_ci.set_ylabel("$\lambda$")
    ax_ci.set_title("Simulations of Stocbio SDE - C.I. $\lambda$-values")
    fig_ci.show()
    
    
    ax2.set_xlabel("time")
    ax2.set_ylabel("$x$")
    ax2.legend()
    ax2.set_title("Simulations of Stocbio SDE - $X$-values")
    fig2.show()
    
    ax2_ci.set_xlabel("time")
    ax2_ci.set_ylabel("$x$")
    ax2_ci.set_title("Simulations of Stocbio SDE - C.I. $X$-values")
    fig2_ci.show()
    
    return TS, xs, ls, xsm, XCIL, XCIU, lsm, LCIL, LCIU

    
def plot_simulations_mckean(num_sims: int, num_xs: int):
    """Plot results of stocbio run alongside confidence intervals"""
    palette_num = get_N_HexCol(num_sims)
    fig, ax = plt.subplots(1)
    fig_ci, ax_ci = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    fig2_ci, ax2_ci = plt.subplots(1)
    #fig_t, ax_t = plt.subplots(1)
    """ Plot several simulations in one image."""
    xs_many = []
    ls_many = []
    for i in range(num_sims):
        TS, xs, ls = run_mckean(num_xs)
        xs_many.append(xs)
        ls_many.append(ls)
        ax.plot(TS, ls, palette_num[i])
        #ax2.plot(TS, np.mean(xs, axis = 0), palette_num[i])
        ax2.plot(TS, xs, palette_num[i])
        #for j in range(num_xs):
        #    ax2.plot(TS, xs[j], palette_num[i])
        
    xsnp = np.asarray(xs_many)
    lsnp = np.asarray(ls_many)
    xsm, XCIL, XCIU, lsm, LCIL, LCIU = get_CI(xsnp,lsnp)
    ax_ci.plot(TS,lsm)
    ax_ci.fill_between(TS, LCIL, LCIU, color="blue", alpha=.15)
    ax2_ci.plot(TS,xsm)
    ax2_ci.fill_between(TS, XCIL, XCIU, color="blue", alpha=.15)

    ax.set_xlabel("time")
    ax.set_ylabel("$\lambda$")
    ax.legend()
    ax.set_title("Simulations of Mckean-Vlasov SDE - $\lambda$-values")
    fig.show()
    
    ax_ci.set_xlabel("time")
    ax_ci.set_ylabel("$\lambda$")
    ax_ci.set_title("Simulations of Mckean-Vlasov SDE - C.I. $\lambda$-values")
    fig_ci.show()
    
    ax2.set_xlabel("time")
    ax2.set_ylabel("$x$")
    ax2.legend()
    ax2.set_title("Simulations of Mckean-Vlasov SDE - $X$-values")
    fig2.show()
    
    ax2_ci.set_xlabel("time")
    ax2_ci.set_ylabel("$x$")
    ax2_ci.set_title("Simulations of Mckean-Vlasov SDE - C.I. $X$-values")
    fig2_ci.show()
    
    return TS, xs, ls, xsm, XCIL, XCIU, lsm, LCIL, LCIU


if __name__ == "__main__":
    NUM_SIMS = 10
    mckean_samples = 100
    TS, xs, ls,       xsm1, XCIL1, XCIU1, lsm1, LCIL1, LCIU1 = plot_simulations(NUM_SIMS)
    TS_m, xs_m, ls_m, xsm2, XCIL2, XCIU2, lsm2, LCIL2, LCIU2 = plot_simulations_mckean(NUM_SIMS,mckean_samples)
    
    fig_both_x, ax_x = plt.subplots(1)
    fig_both_l, ax_l = plt.subplots(1)
    
    ax_x.plot(TS, xsm1, label = 'stocbio',color="blue")
    ax_x.fill_between(TS, XCIL1, XCIU1, color="blue", alpha=.15)
    ax_x.plot(TS_m, xsm2, label = 'mckean',color="orange")
    ax_x.fill_between(TS_m, XCIL2, XCIU2, color="orange", alpha=.15)
    ax_x.legend()
    ax_x.set_xlabel("time")
    ax_x.set_ylabel("X")
    str1 = "2-D Problem: " + str(NUM_SIMS) + " simulations, " + "95%C.I. X-values"
    ax_x.set_title( str1 )
    fig_both_x.show()
    
    ax_l.plot(TS, lsm1, label = 'stocbio',color="blue")
    ax_l.fill_between(TS, LCIL1, LCIU1, color="blue", alpha=.15)
    ax_l.plot(TS_m, lsm2, label = 'mckean',color="orange")
    ax_l.fill_between(TS_m, LCIL2, LCIU2, color="orange", alpha=.15)
    ax_l.legend()
    str2 = "2-D Problem: " + str(NUM_SIMS) + " simulations, " + "95%C.I. $\lambda$-values"
    ax_l.set_title(str2)
    ax_l.set_xlabel("time")
    ax_l.set_ylabel("$\lambda$")
    fig_both_l.show()