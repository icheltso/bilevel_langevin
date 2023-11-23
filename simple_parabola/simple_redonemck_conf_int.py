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
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import random
import colorsys


class Model:
    """Stochastic model constants."""
    SIGMA_X = 1
    SIGMA_lam = 0.1
    """Epsilon serves to differentiate between the two timesteps of the inner and outer problem"""
    EPSILON = 0.05


def getC(lam):
    return (1/2)*((1/(1+lam**2))-2)**2

def getF(x,lam):
    return (1/2) * (x-1)**2 + (1/2) * lam**2 * x**2

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
    #print(nablamFs)
    #print(xvals)
    #print(integ)
    
    return integ, xvals


def run_sde():
    """ Return the result of one full simulation."""
    T_INIT = 0
    T_END = 10
    eps = Model.EPSILON
    
    N = 10000  # Compute at 1000 grid points
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
    
    N = 10000  # Compute at 1000 grid points
    DT = float(T_END - T_INIT) / N
    TS = np.arange(T_INIT, T_END + DT, DT)


    X_INIT = 0
    L_INIT = 1
    
    cl_sig = 1
    X_CLOUD = np.random.normal(0,cl_sig,num_xs)
    X_CLOUD_ALL = np.zeros((TS.size,num_xs))
    X_CLOUD_ALL[0,:] = X_CLOUD

    """Initialize arrays to save intermediate results"""
    #xs = np.zeros((num_xs,TS.size))
    xs = np.zeros(TS.size)
    #xs[:,0] = X_INIT
    xs[0] = X_INIT
    ls = np.zeros(TS.size)
    ls[0] = L_INIT
    mckean_track = np.zeros(TS.size)
    for i in range(1, TS.size):
        t = T_INIT + (i - 1) * DT
        lam = ls[i - 1]
        x = xs[i-1]
        """Generate multiple X values for next timestep alongside the integral term"""
        integ, X_CLOUD_ALL[i,:] = mckean_term(X_CLOUD_ALL[i-1,:], lam, t, DT, num_xs)
        #X_CLOUD_ALL[i,:] = X_CLOUD
        #xs[:,i] = xvs
        #x = xvs[x_arg]
        xs[i] = x + mu_x(x, lam, t) * DT/eps + sigma_x(x, lam, t) * dW(DT)
        mvt = (-1/eps) * nablam_F(x,lam,t) + (1/eps)*integ
        mckean_track[i] = integ - nablam_F(x,lam,t)
        ls[i] = lam + mvt * DT + mu_lam(x, lam, t) * DT + sigma_lam(x, lam, t) * dW(DT)
        
        

    return TS, xs, ls, X_CLOUD_ALL, mckean_track

"""Generate colour palette with n colours (For mckean plots)"""
def get_N_HexCol(N):
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

def get_CI(xs):
    """Generate 95% confidence interval for graphs"""
    pt_ct = np.size(xs,0)
    xsm = np.mean(xs, axis = 0)
    xstd = np.std(xs, axis = 0)
    
    CI = 1.96 * xstd / np.sqrt(pt_ct)
    #XCI = xstd
    CIL = xsm - CI
    CIU = xsm + CI
    
    
    return xsm, CIL, CIU

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
    fig_xtra, ax_xtra = plt.subplots(1)
    fig_dens, ax_dens = plt.subplots(1,2,tight_layout=True)
    """ Plot several simulations in one image."""
    xs_many = []
    ls_many = []
    for i in range(num_sims):
        print("Started iteration " + str(i+1) + " of " + str(num_sims))
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
    
    """Fetch the last iterate from every run - needed to compare histograms to expected density."""
    last_xs = xsnp[:,-1]
    last_ls = lsnp[:,-1]
    """Obtain expected densities."""
    p_x_l, p_l = compare_distrib(last_xs,last_ls)
    """sort x and lambda with their densities. Otherwise lineplot comes out weird."""
    #xs, p_x_l = zip(*sorted(zip(xs, p_x_l)))
    #ls, p_l = zip(*sorted(zip(ls, p_l)))
    """Define number of bins for histogram"""
    n_bins = 30
    ax_dens[0].hist(last_xs, bins=n_bins, density = True, label = 'real')
    ax_dens[0].plot(last_xs, p_x_l, 'o', label = 'expected')
    ax_dens[1].hist(last_ls, bins=n_bins, density = True, label = 'real')
    ax_dens[1].plot(last_ls, p_l,'o', label = 'expected')
    
    ax_dens[0].set_xlabel("X")
    ax_dens[0].set_ylabel("$p(x|\lambda)$")
    ax_dens[1].set_xlabel("$\lambda$")
    ax_dens[1].set_ylabel("$p(\lambda)$")
    ax_dens[0].legend()
    ax_dens[1].legend()
    ax_dens[0].set_title("Stocbio - X densities")
    ax_dens[1].set_title("Stocbio - $\lambda$ densities")
    fig_dens.show()
    
    xsm, XCIL, XCIU = get_CI(xsnp)
    lsm, LCIL, LCIU = get_CI(lsnp)
    
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
    fig3_ci, ax3_ci = plt.subplots(1)
    fig_cloud, ax_cloud = plt.subplots(1)
    fig_xtra, ax_xtra = plt.subplots(1)
    fig_dens, ax_dens = plt.subplots(1,2,tight_layout=True)
    """ Plot several simulations in one image."""
    """Create storage for data over multiple runs"""
    xs_many = []
    ls_many = []
    cl_many = []
    cl_many_L = []
    cl_many_U = []
    for i in range(num_sims):
        print("Started iteration " + str(i+1) + " of " + str(num_sims))
        TS, xs, ls, x_cloud, extra_term = run_mckean(num_xs)
        xs_many.append(xs)
        ls_many.append(ls)
        cloud_m, cloud_L, cloud_U = get_CI(x_cloud.transpose())
        #cl_many.append(cloud_m)
        #cl_many_L.append(cloud_L)
        #cl_many_U.append(cloud_U)
        ax.plot(TS, ls, palette_num[i])
        ax2.plot(TS, xs, palette_num[i])
        ax3_ci.plot(TS,cloud_m)
        ax3_ci.fill_between(TS, cloud_L, cloud_U, color=palette_num[i], alpha=.15)
        ax_cloud.loglog(TS,np.std(x_cloud.transpose(), axis = 0)**2,color=palette_num[i])
        ax_xtra.loglog(TS,np.abs(extra_term),color=palette_num[i])
        
    xsnp = np.asarray(xs_many)
    lsnp = np.asarray(ls_many)
    
    """Fetch the last iterate from every run - needed to compare histograms to expected density."""
    last_xs = xsnp[:,-1]
    last_ls = lsnp[:,-1]
    """Obtain expected densities."""
    p_x_l, p_l = compare_distrib(last_xs,last_ls)
    print(last_xs)
    print(p_x_l)
    print(last_ls)
    print(p_l)
    """sort x and lambda with their densities. Otherwise lineplot comes out weird."""
    #xs, p_x_l = zip(*sorted(zip(xs, p_x_l)))
    #ls, p_l = zip(*sorted(zip(ls, p_l)))
    """Define number of bins for histogram"""
    n_bins = 30
    
    ax_dens[0].hist(last_xs, bins=n_bins, density = True, label = 'real')
    ax_dens[0].plot(last_xs, p_x_l, 'o', label = 'expected')
    ax_dens[1].hist(last_ls, bins=n_bins, density = True, label = 'real')
    ax_dens[1].plot(last_ls, p_l, 'o', label = 'expected')
    
    ax_dens[0].set_xlabel("X")
    ax_dens[0].set_ylabel("$p(x|\lambda)$")
    ax_dens[1].set_xlabel("$\lambda$")
    ax_dens[1].set_ylabel("$p(\lambda)$")
    ax_dens[0].legend()
    ax_dens[1].legend()
    ax_dens[0].set_title("M-V - X densities")
    ax_dens[1].set_title("M-V - $\lambda$ densities")
    fig_dens.show()
    
    
    xsm, XCIL, XCIU = get_CI(xsnp)
    lsm, LCIL, LCIU = get_CI(lsnp)
    """Plot confidence intervals for x, lambda and the cloud"""
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
    ax2.set_ylabel("$X$")
    ax2.legend()
    ax2.set_title("Simulations of Mckean-Vlasov SDE - $X$-values")
    fig2.show()
    
    ax2_ci.set_xlabel("time")
    ax2_ci.set_ylabel("$X$")
    ax2_ci.set_title("Simulations of Mckean-Vlasov SDE - C.I. $X$-values")
    fig2_ci.show()
    
    ax3_ci.set_xlabel("time")
    ax3_ci.set_ylabel("$X^j$")
    ax3_ci.set_title("Simulations of Mckean-Vlasov SDE - Cloud C.I. $X$-values")
    fig3_ci.show()
    
    ax_cloud.set_xlabel("time")
    ax_cloud.set_ylabel("$(\sigma(X^j))^2$")
    ax_cloud.set_title("Simulations of Mckean-Vlasov SDE - Variance of Cloud")
    fig_cloud.show()
    
    ax_xtra.set_xlabel("time")
    ax_xtra.set_ylabel(r"$|\int \nabla_\lambda F \,\mu_t(dx\mid\lambda) - \nabla_{\lambda} F |$")
    ax_xtra.set_title("Simulations of Mckean-Vlasov SDE - Extra term size")
    fig_xtra.show()
    
    return TS, xs, ls, xsm, XCIL, XCIU, lsm, LCIL, LCIU

def get_Zlam(lam):
    beta_x = 2 / (Model.SIGMA_X)**2
    eps = Model.EPSILON
    Zlam = np.sqrt(2*eps*np.pi/(beta_x*(lam**2 + 1))) * np.exp( -beta_x/(2*eps) * (1 - (1/(1+lam**2)) ))
    return Zlam

def compare_distrib(x,lam):
    """Take arrays of x and lambda over NUM_SIMS simulations at the same timestep."""
    """Return expected distribution curves for x and lam."""
    no_pts = np.size(x)
    beta_x = 2 / (Model.SIGMA_X)**2
    beta_lam = 2 / (Model.SIGMA_lam)**2
    eps = Model.EPSILON
    Z_out, Z_err = integrate.quad(lambda l: np.exp(-(beta_lam/2) * (1 / (1+l**2) - 2)**2 ), -np.inf, np.inf)
    print(Z_out)
    print(Z_err)
    p_x_l = np.zeros(no_pts)
    p_l = np.zeros(no_pts)
    for i in range(no_pts):
        xcur = x[i]
        lcur = lam[i]
        Z_in = get_Zlam(lcur)
        p_x_l[i] = (1 / Z_in) * np.exp((-beta_x / eps) * getF(xcur,lcur))
        p_l[i] = (1 / Z_out) * np.exp((-beta_lam) * getC(lcur))
        
    return p_x_l, p_l
    
    


if __name__ == "__main__":
    NUM_SIMS = 300
    mckean_samples = 100
    print("Starting Stocbio")
    TS, xs, ls,       xsm1, XCIL1, XCIU1, lsm1, LCIL1, LCIU1 = plot_simulations(NUM_SIMS)
    print("Starting M-V")
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