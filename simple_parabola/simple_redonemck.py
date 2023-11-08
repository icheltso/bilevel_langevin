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
    SIGMA_X = 0.06
    SIGMA_lam = 0.06
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
    T_END = 10
    eps = Model.EPSILON
    
    N = 4000  # Compute at 1000 grid points
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
    T_END = 10
    eps = Model.EPSILON
    
    N = 4000  # Compute at 1000 grid points
    DT = float(T_END - T_INIT) / N
    TS = np.arange(T_INIT, T_END + DT, DT)


    X_INIT = 0
    L_INIT = 1

    """Initialize arrays to save intermediate results"""
    #xs = np.zeros((num_xs,TS.size))
    xs = np.zeros(TS.size)
    #xs[:,0] = X_INIT
    xs[0] = X_INIT
    ls = np.zeros(TS.size)
    ls[0] = L_INIT
    """Initialize first batch of x-values"""
    xvs = X_INIT * np.ones(TS.size)
    x_arg = random.choice(range(num_xs))
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
    palette_num = get_N_HexCol(num_sims)
    fig, ax = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    """ Plot several simulations in one image."""
    for i in range(num_sims):
        TS, xs, ls = run_sde()
        ax.plot(TS, ls, palette_num[i])
        #ax2.plot(TS, np.mean(xs, axis = 0), palette_num[i])
        ax2.plot(TS, xs, palette_num[i])
        #for j in range(num_xs):
        #    ax2.plot(TS, xs[j], palette_num[i])
    ax.set_xlabel("time")
    ax.set_ylabel("$\lambda$")
    ax.legend()
    ax.set_title("Simulations of Stocbio SDE - $\lambda$-values")
    fig.show()
    
    ax2.set_xlabel("time")
    ax2.set_ylabel("$x$")
    ax2.legend()
    ax2.set_title("Simulations of Stocbio SDE - $X$-values")
    fig2.show()
    
    return TS, xs, ls

    
def plot_simulations_mckean(num_sims: int, num_xs: int):
    palette_num = get_N_HexCol(num_sims)
    fig, ax = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    """ Plot several simulations in one image."""
    for i in range(num_sims):
        TS, xs, ls = run_mckean(num_xs)
        ax.plot(TS, ls, palette_num[i])
        #ax2.plot(TS, np.mean(xs, axis = 0), palette_num[i])
        ax2.plot(TS, xs, palette_num[i])
        #for j in range(num_xs):
        #    ax2.plot(TS, xs[j], palette_num[i])
        
        

    ax.set_xlabel("time")
    ax.set_ylabel("$\lambda$")
    ax.legend()
    ax.set_title("Simulations of Mckean-Vlasov SDE - $\lambda$-values")
    fig.show()
    
    ax2.set_xlabel("time")
    ax2.set_ylabel("$x$")
    ax2.legend()
    ax2.set_title("Simulations of Mckean-Vlasov SDE - $X$-values")
    fig2.show()
    
    return TS, xs, ls


if __name__ == "__main__":
    NUM_SIMS = 10
    mckean_samples = 100
    TS, xs, ls = plot_simulations(NUM_SIMS)
    TS_m, xs_m, ls_m = plot_simulations_mckean(NUM_SIMS,mckean_samples)