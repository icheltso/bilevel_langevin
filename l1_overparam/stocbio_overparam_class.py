# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:17:03 2023

Solve bilevel problem as system of SDEs:
    min_lam  C(x(lam),lam)
    s.t. x(lam) = argmin_x F(x,lam)
2-D PROBLEM:
    C(x,lam) = (1/2) * (x)^2 + (1/2) * (lambda)^2 + (1/2)*G(x*lambda)
    F(x,lam) = (1/2) * (x)^2 + (1/2) * (lambda)^2 + (1/2)*G(x*lambda)
    for some smooth G: R -> R
    
This is equivalent to solving 

    min_z 2|z| + G(z)
    
    In our case choose G(t) = (t-2)**2
    
    Expected optimum at z = 1 or, equivalently, at x = lam = +-1.
    
    

@author: ichel
"""
import os

import math
import numpy as np
import scipy.stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import random
import colorsys
import pickle as pkl

subfolder_name = "SIMULATIONS"
current_directory = os.getcwd()
sim_path = os.path.join(current_directory, subfolder_name)

class Model_l1:
    def __init__(self,TOTAL_SIMS,SIGMA,EPS,F_SCALE,C_SCALE,NUM_CLOUD):
        """Stochastic model constants."""
        self.SIGMA_X = SIGMA
        self.SIGMA_lam = SIGMA
        """In order to dampen the noise without changing the invariant measure, we can scale F and C."""
        """The noise for the inner SDE should be proportional to 1/sqrt(F_SCALE)"""
        self.F_SCALE = F_SCALE
        self.C_SCALE = C_SCALE
        """Size of cloud for approximating the extra mckean term"""
        self.NUM_CLOUD = NUM_CLOUD
        
        """Runtime-specific parameters"""
        self.T_END = 12
        self.GRID = 4800
        """Epsilon serves to differentiate between the two timesteps of the inner and outer problem"""
        self.EPSILON = EPS
    

    def G_xl(self,t: float) -> float:
        """Encode G(x,lambda)"""
        #gxl = np.exp(x*lam)
        gxl = (1/2)*(t-2)**2
        return gxl
    
    def grad_G_xl(self,t: float) -> float:
        """Encode derivative of G(t)"""
        #gxl = np.exp(x*lam)
        gxl = (t-2)
        return gxl
    
    def hess_G_xl(self,t: float) -> float:
        """Encode second derivative of G(x,lambda)"""
        #gxl = np.exp(x*lam)
        gxl = 1
        return gxl

    def mu_x(self,x: float, lam: float, _t: float) -> float:
        """Implement the drift term for x (inner, quick sde)."""
        nab_x_F = x + lam * self.grad_G_xl(x*lam)
        return nab_x_F

    def mu_lam(self,x: float, lam: float, _t: float) -> float:
        """Implement the drift term for lambda (outer, slow sde)."""
        nab_l_C = lam + x * self.grad_G_xl(x*lam)
        nab_xl_F = self.grad_G_xl(x*lam) + x * lam * self.hess_G_xl(x*lam)
        nab_xx_F = 1 + (lam**2)*self.hess_G_xl(x*lam)
        nab_x_C = x + lam * self.grad_G_xl(x*lam)
        
        nab_L = nab_l_C + (nab_xl_F/nab_xx_F) * nab_x_C
    
        return nab_L


    def sigma_x(self,x: float, lam: float, _t: float) -> float:
        """Implement the diffusion coefficient for x (inner problem)."""
        return self.SIGMA_X

    def sigma_lam(self,x: float, lam: float, _t: float) -> float:
        """Implement the diffusion coefficient for lambda (outer problem)."""
        return self.SIGMA_lam


    def dW(self,delta_t: float) -> float:
        """Sample a random number at each call."""
        return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

    def nablam_F(self,x: float, lam: float, _t: float) -> float:
        """Return the gradient of F w.r.t. lambda"""
        return lam + x * self.grad_G_xl(x*lam)

    def mckean_term(self,xvs, lam: float, t: float, DT: float, num_xs: int) -> float:
        """Return the extra term from the Mckean-Vlasov SDE"""
    
        xvals = np.zeros(num_xs)
        nablamFs = np.zeros(num_xs)
        pdfx = np.zeros(num_xs)
        integ = 0
        eps0 = self.EPSILON
    
        for i in range(num_xs):
            xvals[i] = xvs[i] - self.mu_x(xvs[i],lam,t)*DT/eps0 + self.sigma_x(xvs[i], lam, t) * self.dW(DT)
            nablamFs[i] = self.nablam_F(xvals[i],lam,t)
            integ += nablamFs[i]
        
        integ = integ/num_xs
    
        return integ, xvals


    def run_sde(self):
        """ Return the result of one full simulation."""
        T_INIT = 0
        T_END = self.T_END
        eps = self.EPSILON
    
        N = self.GRID  # Compute at 1000 grid points
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
            xs[i] = x - self.mu_x(x, lam, t) * DT/eps + self.sigma_x(x, lam, t) * self.dW(DT)
            ls[i] = lam - self.mu_lam(x, lam, t) * DT + self.sigma_lam(x, lam, t) * self.dW(DT)
            if np.abs(xs[i]) > 1e5 or np.abs(ls[i]) > 1e5:
                return math.nan, math.nan, math.nan
        

        return TS, xs, ls

    def run_mckean(self):
        num_xs = self.NUM_CLOUD
        """ Return the result of one full simulation."""
        T_INIT = 0
        T_END = self.T_END
        eps = self.EPSILON
        
        N = self.GRID  # Compute at 1000 grid points
        DT = float(T_END - T_INIT) / N
        TS = np.arange(T_INIT, T_END + DT, DT)
        

        X_INIT = 0
        L_INIT = 1
    
        cl_sig = 1
        X_CLOUD = np.random.normal(0,cl_sig,num_xs)
        X_CLOUD_ALL = np.zeros((TS.size,num_xs))
        X_CLOUD_ALL[0,:] = X_CLOUD

        """Initialize arrays to save intermediate results"""
        xs = np.zeros(TS.size)
        xs[0] = X_INIT
        ls = np.zeros(TS.size)
        ls[0] = L_INIT
        mckean_track = np.zeros(TS.size)
        for i in range(1, TS.size):
            t = T_INIT + (i - 1) * DT
            lam = ls[i - 1]
            x = xs[i-1]
            """Generate multiple X values for next timestep alongside the integral term"""
            integ, X_CLOUD_ALL[i,:] = self.mckean_term(X_CLOUD_ALL[i-1,:], lam, t, DT, num_xs)
            xs[i] = x - self.mu_x(x, lam, t) * DT/eps + self.sigma_x(x, lam, t) * self.dW(DT)
            mvt = (-1/eps) * self.nablam_F(x,lam,t) + (1/eps)*integ
            mckean_track[i] = integ - self.nablam_F(x,lam,t)
            ls[i] = lam + mvt * DT - self.mu_lam(x, lam, t) * DT + self.sigma_lam(x, lam, t) * self.dW(DT)
            if np.abs(xs[i]) > 1e5 or np.abs(ls[i]) > 1e5:
                return math.nan, math.nan, math.nan, math.nan, math.nan
        

        return TS, xs, ls, X_CLOUD_ALL, mckean_track
    
    def compare_distrib(self,z):
        """Return expected distribution curve for z = x*lam."""
        no_pts = np.size(z)
        #xlam = x * lam
        beta = 2 / (self.SIGMA_X)**2
        #beta_xl = 1
        #beta_xl = beta_x + beta_lam
        eps = self.EPSILON
        """Obtain normalizing constant"""
        Z_out, Z_err = integrate.quad(lambda z: np.exp(-beta*(np.abs(z) + self.G_xl(z))), -np.inf, np.inf)
        print(Z_out)
        print(Z_err)
        p_x_l = np.zeros(no_pts)
        for i in range(no_pts):
            #xlcur = xlam[i]
            zcur = z[i]
            p_x_l[i] = (1 / Z_out) * np.exp(-beta*(np.abs(zcur) + self.G_xl(zcur)))
            
        return p_x_l

def get_CI(xs):
    """Generate 95% confidence interval for graphs"""
    pt_ct = np.size(xs,0)
    xsm = np.mean(xs, axis = 0)
    xstd = np.std(xs, axis = 0)
    
    #CI = 1.96 * xstd / np.sqrt(pt_ct)
    CI = xstd
    #XCI = xstd
    CIL = xsm - CI
    CIU = xsm + CI
    
    
    return xsm, CIL, CIU

"""Generate colour palette with n colours (For mckean plots)"""
def get_N_HexCol(N):
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out


def plot_simulations(model,num_sims: int):
    """Plot results of stocbio run alongside confidence intervals"""
    palette_num = get_N_HexCol(num_sims)
    fig, ax = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    fig_dens, ax_dens = plt.subplots(1)
    """ Plot several simulations in one image."""
    xs_many = []
    ls_many = []
    for i in range(num_sims):
        print("Started iteration " + str(i+1) + " of " + str(num_sims) + ". sigma = " + str(model.SIGMA_X) + ", k_F = " + str(model.F_SCALE))
        TS, xs, ls = model.run_sde()
        if np.isnan(TS).any() == False:
            xs_many.append(xs)
            ls_many.append(ls)
            ax.plot(TS, ls, palette_num[i])
            #ax2.plot(TS, np.mean(xs, axis = 0), palette_num[i])
            ax2.plot(TS, xs, palette_num[i])
            #for j in range(num_xs):
            #    ax2.plot(TS, xs[j], palette_num[i])
        
    xsnp = np.asarray(xs_many)
    lsnp = np.asarray(ls_many)
    
    """Get only the positive/negative values of x/lam. For C.I. plotting"""
    #x_pos = xsnp[xsnp>=0]
    #x_neg = xsnp[xsnp<0]
    #l_pos = lsnp[lsnp>=0]
    #l_neg = lsnp[lsnp<0]
    
    """Fetch the last iterate from every run - needed to compare histograms to expected density."""
    last_xs = xsnp[:,-1]
    last_ls = lsnp[:,-1]
    last_xl = last_xs * last_ls
    
    min_z = np.min(last_xl)
    max_z = np.max(last_xl)
    zstd = np.std(last_xl)

    z_for_dens = np.linspace(min_z-zstd,max_z+zstd,1000)
    """Obtain expected densities."""
    p_x_l = model.compare_distrib(z_for_dens)
    print(p_x_l)
    """sort x and lambda with their densities. Otherwise lineplot comes out weird."""
    #xs, p_x_l = zip(*sorted(zip(xs, p_x_l)))
    #ls, p_l = zip(*sorted(zip(ls, p_l)))
    """Define number of bins for histogram"""
    n_bins = 30
    
    ax_dens.hist(last_xl, bins=n_bins, density = True, label = 'numeric')
    ax_dens.plot(z_for_dens, p_x_l, label = 'exact')
    ax_dens.set_xlabel("z = x * $\lambda$")
    ax_dens.set_ylabel("$p(z)$")
    ax_dens.legend()
    ax_dens.set_title("Stocbio - Z densities")
    fig_dens.show()
    
    xsm, XCIL, XCIU = get_CI(xsnp)
    lsm, LCIL, LCIU = get_CI(lsnp)
    
    
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
    
    """Save plots"""
    dirstr = "sigma_" + str(model.SIGMA_X) + "_k_F_" + str(model.F_SCALE) + "_k_C_" + str(model.C_SCALE) + "_num_sims_" + str(num_sims)
    subfolder_path = os.path.join(sim_path, dirstr)
    iterates_path = os.path.join(subfolder_path, "ALL_ITERATES")
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    if not os.path.exists(iterates_path):
        os.makedirs(iterates_path)
    fig.savefig(iterates_path + '/stocbio_lam.png')
    fig2.savefig(iterates_path + '/stocbio_x.png')
    fig_dens.savefig(subfolder_path + '/stocbio_density.png')
    
    file_path = subfolder_path + '/data_sbio.pkl'
    with open(file_path, 'wb') as f:
        pkl.dump([TS, xsnp, lsnp],f)
        
    
    return TS, xsnp, lsnp

    
def plot_simulations_mckean(model,num_sims: int):
    """Plot results of stocbio run alongside confidence intervals"""
    palette_num = get_N_HexCol(num_sims)
    fig, ax = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    fig_xtra, ax_xtra = plt.subplots(1)
    fig_dens, ax_dens = plt.subplots(1)
    """ Plot several simulations in one image."""
    """Create storage for data over multiple runs"""
    xs_many = []
    ls_many = []
    cld_var = []
    int_term = []
    for i in range(num_sims):
        print("Started iteration " + str(i+1) + " of " + str(num_sims))
        TS, xs, ls, x_cloud, extra_term = model.run_mckean()
        if np.isnan(TS).any() == False:
            xs_many.append(xs)
            ls_many.append(ls)
            cld_var.append(np.std(x_cloud.transpose(), axis = 0)**2)
            int_term.append(extra_term)
            ax.plot(TS, ls, palette_num[i])
            ax2.plot(TS, xs, palette_num[i])
            ax_xtra.loglog(TS,np.abs(extra_term),color=palette_num[i])
        
    xsnp = np.asarray(xs_many)
    lsnp = np.asarray(ls_many)
    
    """Get only the positive/negative values of x/lam. For C.I. plotting"""
    #x_pos = xsnp[xsnp>=0]
    #x_neg = xsnp[xsnp<0]
    #l_pos = lsnp[lsnp>=0]
    #l_neg = lsnp[lsnp<0]
    
    """Fetch the last iterate from every run - needed to compare histograms to expected density."""
    last_xs = xsnp[:,-1]
    last_ls = lsnp[:,-1]
    last_xl = last_xs * last_ls
    min_z = np.min(last_xl)
    max_z = np.max(last_xl)
    zstd = np.std(last_xl)

    z_for_dens = np.linspace(min_z-zstd,max_z+zstd,1000)
    """Obtain expected densities."""
    p_x_l = model.compare_distrib(z_for_dens)
    print(p_x_l)
    """sort x and lambda with their densities. Otherwise lineplot comes out weird."""
    #xs, p_x_l = zip(*sorted(zip(xs, p_x_l)))
    #ls, p_l = zip(*sorted(zip(ls, p_l)))
    """Define number of bins for histogram"""
    n_bins = 30
    
    ax_dens.hist(last_xl, bins=n_bins, density = True, label = 'numeric')
    ax_dens.plot(z_for_dens, p_x_l, label = 'exact')
    ax_dens.set_xlabel("z = x * $\lambda$")
    ax_dens.set_ylabel("$p(z)$")
    ax_dens.legend()
    ax_dens.set_title("M-V - Z densities")
    fig_dens.show()
    
    """Obtain 95% confidence intervals"""
    #xsp, XCILp, XCIUp = get_CI(x_pos)
    #lsp, LCILp, LCIUp = get_CI(l_pos)
    #xsn, XCILn, XCIUn = get_CI(x_neg)
    #lsn, LCILn, LCIUn = get_CI(l_neg)
    

    ax.set_xlabel("time")
    ax.set_ylabel("$\lambda$")
    ax.legend()
    ax.set_title("Simulations of Mckean-Vlasov SDE - $\lambda$-values")
    fig.show()
    
    ax2.set_xlabel("time")
    ax2.set_ylabel("$X$")
    ax2.legend()
    ax2.set_title("Simulations of Mckean-Vlasov SDE - $X$-values")
    fig2.show()
    
    ax_xtra.set_xlabel("time")
    ax_xtra.set_ylabel(r"$|\int \nabla_\lambda F \,\mu_t(dx\mid\lambda) - \nabla_{\lambda} F |$")
    ax_xtra.set_title("Simulations of Mckean-Vlasov SDE - Extra term size")
    fig_xtra.show()
    
    dirstr = "sigma_" + str(model.SIGMA_X) + "_k_F_" + str(model.F_SCALE) + "_k_C_" + str(model.C_SCALE) + "_num_sims_" + str(num_sims)
    subfolder_path = os.path.join(sim_path, dirstr)
    iterates_path = os.path.join(subfolder_path, "ALL_ITERATES")
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    if not os.path.exists(iterates_path):
        os.makedirs(iterates_path)
    fig.savefig(iterates_path + '/M_V_lam.png')
    fig2.savefig(iterates_path + '/M_V_x.png')
    fig_dens.savefig(subfolder_path + '/M_V_density.png')
    fig_xtra.savefig(subfolder_path + '/M_V_extra.png')
    
    file_path = subfolder_path + '/data_MV.pkl'
    with open(file_path, 'wb') as f:
        pkl.dump([TS, xsnp, lsnp, cld_var, int_term], f)
    
    return TS, xsnp, lsnp
    
    


if __name__ == "__main__":
    TOTAL_SIMS = 1000
    SIGMA_ARR = [0.1, 0.5, 1]
    EPS = 0.05
    F_SCALE_ARR = [0.1, 0.5, 1]
    C_SCALE = 1
    NUM_CLOUD = 100
    
    for SIGMA in SIGMA_ARR:
        for F_SCALE in F_SCALE_ARR:
            print("Started for sigma = " + str(SIGMA) + ", k_F = " + str(F_SCALE))
            model = Model_l1(TOTAL_SIMS, SIGMA, EPS, F_SCALE, C_SCALE, NUM_CLOUD)
            print("Starting Stocbio")
            TS, xs, ls = plot_simulations(model,TOTAL_SIMS)
            print("Starting M-V")
            TS_m, xs_m, ls_m = plot_simulations_mckean(model,TOTAL_SIMS)
    
            z1 = xs * ls
            z2 = xs_m * ls_m
            
            z1m, z1L, z1U = get_CI(z1)
            z2m, z2L, z2U = get_CI(z2)
            
            fig_z, ax_z = plt.subplots(1)
            
            ax_z.plot(TS, z1m, label = 'stocbio',color="blue")
            ax_z.fill_between(TS, z1L, z1U, color="blue", alpha=.15)
            ax_z.plot(TS_m, z2m, label = 'mckean',color="orange")
            ax_z.fill_between(TS_m, z2L, z2U, color="orange", alpha=.15)
            ax_z.legend()
            ax_z.set_xlabel("time")
            ax_z.set_ylabel("Z")
            str1 = "Overparametrized Problem: " + str(TOTAL_SIMS) + " simulations, " + "+-1 s.t.d. C.I. Z-values"
            ax_z.set_title( str1 )
            fig_z.show()
    
    
            dirstr = "sigma_" + str(model.SIGMA_X) + "_k_F_" + str(model.F_SCALE) + "_k_C_" + str(model.C_SCALE) + "_num_sims_" + str(TOTAL_SIMS)
            subfolder_path = os.path.join(sim_path, dirstr)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
            fig_z.savefig(subfolder_path + '/overparam_z.png')
    
    #fig_both_x, ax_x = plt.subplots(1)
    #fig_both_l, ax_l = plt.subplots(1)
    
    #ax_x.plot(TS, xsm1, label = 'stocbio',color="blue")
    #ax_x.fill_between(TS, XCIL1, XCIU1, color="blue", alpha=.15)
    #ax_x.plot(TS_m, xsm2, label = 'mckean',color="orange")
    #ax_x.fill_between(TS_m, XCIL2, XCIU2, color="orange", alpha=.15)
    #ax_x.legend()
    #ax_x.set_xlabel("time")
    #ax_x.set_ylabel("X")
    #str1 = "2-D Problem: " + str(NUM_SIMS) + " simulations, " + "95%C.I. X-values"
    #ax_x.set_title( str1 )
    #fig_both_x.show()
    
    #ax_l.plot(TS, lsm1, label = 'stocbio',color="blue")
    #ax_l.fill_between(TS, LCIL1, LCIU1, color="blue", alpha=.15)
    #ax_l.plot(TS_m, lsm2, label = 'mckean',color="orange")
    #ax_l.fill_between(TS_m, LCIL2, LCIU2, color="orange", alpha=.15)
    #ax_l.legend()
    #str2 = "2-D Problem: " + str(NUM_SIMS) + " simulations, " + "95%C.I. $\lambda$-values"
    #ax_l.set_title(str2)
    #ax_l.set_xlabel("time")
    #ax_l.set_ylabel("$\lambda$")
    #fig_both_l.show()