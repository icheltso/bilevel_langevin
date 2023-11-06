# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:17:03 2023

Solve bilevel problem as system of SDEs:
    min_lam  C(x(lam),lam)
    s.t. x(lam) = argmin_x F(x,lam)
2-D PROBLEM:
    C(x,lam) = (1/2) * (x)^2 + (1/2) * (lambda)^2 + G(x*lambda)
    F(x,lam) = (1/2) * (x)^2 + (1/2) * (lambda)^2 + G(x*lambda)
    for some smooth G: R -> R
    
This is equivalent to solving 

    min_z |z| + G(z)
    
    In our case choose G(t) = exp(t)
    
    optimal lambda = 0

@author: ichel
"""

import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


class Model:
    """Stochastic model constants."""
    SIGMA_X = 0.06
    SIGMA_lam = 0.06
    """Epsilon serves to differentiate between the two timesteps of the inner and outer problem"""
    EPSILON = 0.05

def G_xl(x: float, lam: float) -> float:
    """Encode G(x,lambda)"""
    #gxl = np.exp(x*lam)
    gxl = (x-1)**2
    return gxl
    
def grad_G_xl(x: float, lam: float) -> float:
    """Encode derivative of G(x,lambda)"""
    #gxl = np.exp(x*lam)
    gxl = 2*(x-1)
    return gxl
    
def hess_G_xl(x: float, lam: float) -> float:
    """Encode second derivative of G(x,lambda)"""
    #gxl = np.exp(x*lam)
    gxl = 2
    return gxl

def mu_x(x: float, lam: float, _t: float) -> float:
    """Implement the drift term for x (inner, quick sde)."""
    nab_x_F = x + lam * grad_G_xl(x,lam)
    return nab_x_F

def mu_lam(x: float, lam: float, _t: float) -> float:
    """Implement the drift term for lambda (outer, slow sde)."""
    nab_l_C = lam + x * grad_G_xl(x,lam)
    nab_xl_F = grad_G_xl(x,lam) + x * lam * hess_G_xl(lam,x)
    nab_xx_F = 1 + (lam**2)*hess_G_xl(lam,x)
    nab_x_C = x + lam * grad_G_xl(x,lam)
    
    nab_L = nab_l_C + (nab_xl_F/nab_xx_F) * nab_x_C
    
    return nab_L


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

def mckean_term(x: float, lam: float, t: float, DT: float, num_xs: int) -> float:
    """Return the extra term from the Mckean-Vlasov SDE"""
    eps0 = Model.EPSILON
    
    mean_X = x + mu_x(x, lam, t)*DT/eps0
    sig_X = sigma_x(x, lam, t) * np.sqrt(DT)
    distrib = scipy.stats.norm(mean_X, sig_X)
    
    xvals = np.zeros(num_xs)
    nablamFs = np.zeros(num_xs)
    pdfx = np.zeros(num_xs)
    integ = 0
    
    for i in range(num_xs):
        xvals[i] = mean_X + sigma_x(x, lam, t) * dW(DT)
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
    
    mvt = (-1/eps0) * nablam_F(x,lam,t) + (1/eps0)*integ
    return mvt


def run_sde():
    """ Return the result of one full simulation."""
    T_INIT = 0
    T_END = 20
    eps = Model.EPSILON
    
    N = 8000  # Compute at 1000 grid points
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
        xs[i] = x - mu_x(x, lam, t) * DT/eps + sigma_x(x, lam, t) * dW(DT)
        ls[i] = lam - mu_lam(x, lam, t) * DT + sigma_lam(x, lam, t) * dW(DT)
        
        

    return TS, xs, ls

def run_mckean(num_xs: int):
    """ Return the result of one full simulation."""
    T_INIT = 0
    T_END = 5
    eps = Model.EPSILON
    
    N = 2000  # Compute at 1000 grid points
    DT = float(T_END - T_INIT) / N
    TS = np.arange(T_INIT, T_END + DT, DT)


    X_INIT = 0
    L_INIT = 1

    xs = np.zeros(TS.size)
    xs[0] = X_INIT
    ls = np.zeros(TS.size)
    ls[0] = L_INIT
    mts = np.zeros(TS.size)
    for i in range(1, TS.size):
        t = T_INIT + (i - 1) * DT
        x = xs[i - 1]
        lam = ls[i - 1]
        mts[i] = mckean_term(x, lam, t, DT, num_xs)
        xs[i] = x + mu_x(x, lam, t) * DT/eps + sigma_x(x, lam, t) * dW(DT)
        ls[i] = lam + mts[i] * DT + mu_lam(x, lam, t) * DT + sigma_lam(x, lam, t) * dW(DT)
        
        

    return TS, xs, ls, mts



def plot_simulations(num_sims: int):
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
    
def plot_simulations_mckean(num_sims: int, num_xs: int):
    fig = plt.figure()
    """ Plot several simulations in one image."""
    for i in range(num_sims):
        TS, xs, ls, mts = run_mckean(num_xs)
        if i == num_sims - 1:
            plt.plot(TS, xs, label='x')
            plt.plot(TS, ls, label='$\lambda$')
        else:
            plt.plot(TS, xs)
            plt.plot(TS, ls)
        

    plt.xlabel("time")
    plt.ylabel("x,$\lambda$")
    plt.legend()
    plt.title("Simulations of Mckean-Vlasov SDE")
    plt.show()
    return TS, xs, ls, mts


if __name__ == "__main__":
    NUM_SIMS = 5
    mckean_samples = 10
    TS, xs, ls = plot_simulations(NUM_SIMS)
    #TS_m, xs_m, ls_m, mts = plot_simulations_mckean(NUM_SIMS,mckean_samples)