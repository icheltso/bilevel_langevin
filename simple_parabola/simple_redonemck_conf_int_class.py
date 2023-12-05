# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:17:03 2023

Solve bilevel problem as system of SDEs:
    min_lam  C(x(lam),lam)
    s.t. x(lam) = argmin_x F(x,lam)
2-D TEST PROBLEM:
    C(x,lam) = k_C * (1/2) * (x-2)^2
    F(x,lam) = k_F * (1/2) * (x-1)^2 + (1/2) * lam^2 * x^2
    
    optimal lambda = 0

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

subfolder_name = "SIMULATION"
current_directory = os.getcwd()
sim_path = os.path.join(current_directory, subfolder_name)



class Model:
    """Model contains the necessary parameters/methods to run a single simulation of either Stocbio or M-V sdes."""
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

    def getC(self,lam):
        """Return C(x(lam),lam)"""
        return self.C_SCALE * (1/2)*((1/(1+lam**2))-2)**2

    def getF(self,x,lam):
        """Return F(x,lam)"""
        return self.F_SCALE * ((1/2) * (x-1)**2 + (1/2) * lam**2 * x**2)

    def mu_x(self,x: float, lam: float, _t: float) -> float:
        """Implement the drift term for x (inner, quick sde)."""
        mux = -self.F_SCALE*((x-1) + (lam**2) * x)
        return mux

    def mu_lam(self,x: float, lam: float, _t: float) -> float:
        """Implement the drift term for lambda (outer, slow sde)."""
        nab_l_C = 0
        nab_xl_F = 2 * self.F_SCALE * x * lam
        nab_xx_F = self.F_SCALE * (1 + lam**2)
        nab_x_C = self.C_SCALE * (x-2)
    
        nab_L = nab_l_C + (nab_xl_F/nab_xx_F) * nab_x_C
    
        #mul = 2*lam*x*(1/(1+lam**2))*(x-2)
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
        return self.F_SCALE * lam * x**2

    def mckean_term(self,xvs, lam: float, t: float, DT: float, num_xs: int) -> float:
        """Return the extra term from the Mckean-Vlasov SDE"""
    
        xvals = np.zeros(num_xs)
        nablamFs = np.zeros(num_xs)
        pdfx = np.zeros(num_xs)
        integ = 0
        eps0 = self.EPSILON
    
        for i in range(num_xs):
            xvals[i] = xvs[i] + self.mu_x(xvs[i],lam,t)*DT/eps0 + self.sigma_x(xvs[i], lam, t) * self.dW(DT)
            nablamFs[i] = self.nablam_F(xvals[i],lam,t)
            integ += nablamFs[i]
        
        integ = integ/num_xs

    
        return integ, xvals


    def run_sde(self):
        """ Return the result of one full simulation of Stocbio."""
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
            xs[i] = x + self.mu_x(x, lam, t) * DT/eps + self.sigma_x(x, lam, t) * self.dW(DT)
            ls[i] = lam + self.mu_lam(x, lam, t) * DT + self.sigma_lam(x, lam, t) * self.dW(DT)

        return TS, xs, ls

    def run_mckean(self):
        num_xs = self.NUM_CLOUD
        """ Return the result of one full simulation of Mckean-Vlasov."""
        T_INIT = 0
        T_END = self.T_END
        eps = self.EPSILON
        N = self.GRID
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
            xs[i] = x + self.mu_x(x, lam, t) * DT/eps + self.sigma_x(x, lam, t) * self.dW(DT)
            mvt = (-1/eps) * self.nablam_F(x,lam,t) + (1/eps)*integ
            mckean_track[i] = integ - self.nablam_F(x,lam,t)
            ls[i] = lam + mvt * DT + self.mu_lam(x, lam, t) * DT + self.sigma_lam(x, lam, t) * self.dW(DT)

        return TS, xs, ls, X_CLOUD_ALL, mckean_track
    
    """instance methods below are used for approximating the p.d.f.s for x and lambda."""
    def get_Zlam(self,lam):
        beta_x = 2 / (self.SIGMA_X)**2
        eps = self.EPSILON
        Zlam = np.sqrt(2*eps*np.pi/(self.F_SCALE*beta_x*(lam**2 + 1))) * np.exp( -(self.F_SCALE*beta_x)/(2*eps) * (1 - (1/(1+lam**2)) ))
        return Zlam

    def get_p_x(self,xspace):
        """Given an array of points, return the image of these under the probability density function of x."""
        beta_lam = 2 / (self.SIGMA_lam)**2
        eps = self.EPSILON
        Z_out, Z_err = integrate.quad(lambda l: np.exp(-beta_lam * self.getC(l)), -np.inf, np.inf)
        p_x = np.zeros(len(xspace))
        for i in range(len(xspace)):
            xi = xspace[i]
            p_x[i], p_x_err = integrate.quad(lambda l: (1 / self.get_Zlam(l)) * np.exp(-(beta_lam) * (self.getC(l) + self.getF(xi,l)/eps)), -np.inf, np.inf)
        
        p_x = (1 / Z_out) * p_x
        return p_x
            
    def get_p_l(self,lspace):
        """Given an array of points, return the image of these under the probability density function of lambda."""
        beta_lam = 2 / (self.SIGMA_lam)**2
        eps = self.EPSILON
        Z_out, Z_err = integrate.quad(lambda l: np.exp(-beta_lam * self.getC(l)), -np.inf, np.inf)
        p_l = np.zeros(len(lspace))
        for i in range(len(lspace)):
            li = lspace[i]
            p_l[i] = np.exp((-beta_lam) * self.getC(li))
        p_l = (1 / Z_out) * p_l
        return p_l

"""Generate colour palette with n colours (For mckean plots)"""
def get_N_HexCol(N):
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

def get_CI(xs):
    """Generate +-1st.d. confidence interval for graphs"""
    pt_ct = np.size(xs,0)
    xsm = np.mean(xs, axis = 0)
    xstd = np.std(xs, axis = 0)
    
    #CI = 1.96 * xstd / np.sqrt(pt_ct)
    #XCI = xstd
    CI = xstd
    CIL = xsm - CI
    CIU = xsm + CI
    
    
    return xsm, CIL, CIU

def plot_simulations(model,num_sims: int):
    """Plot results of stocbio run alongside confidence intervals"""
    palette_num = get_N_HexCol(num_sims)
    fig, ax = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    #fig_t, ax_t = plt.subplots(1)
    fig_dens, ax_dens = plt.subplots(1,2,tight_layout=True)
    """ Plot several simulations in one image."""
    xs_many = []
    ls_many = []
    for i in range(num_sims):
        print("Started iteration " + str(i+1) + " of " + str(num_sims) + ". sigma = " + str(model.SIGMA_X) + ", k_F = " + str(model.F_SCALE))
        TS, xs, ls = model.run_sde()
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
    #p_x, p_l = compare_distrib(last_xs,last_ls)
    """sort x and lambda with their densities. Otherwise lineplot comes out weird."""
    #xs, p_x_l = zip(*sorted(zip(xs, p_x_l)))
    #ls, p_l = zip(*sorted(zip(ls, p_l)))
    """Define number of bins for histogram"""
    n_bins = 30
    """Retrieve data about min/max/spread of final iterates"""
    min_x = np.min(last_xs)
    max_x = np.max(last_xs)
    xstd = np.std(last_xs)
    min_l = np.min(last_ls)
    max_l = np.max(last_ls)
    lstd = np.std(last_ls)
    """Obtain expected densities."""
    x_for_dens = np.linspace(min_x-xstd,max_x+xstd,1000)
    l_for_dens = np.linspace(min_l-lstd,max_l+lstd,1000)
    p_x = model.get_p_x(x_for_dens)
    p_l = model.get_p_l(l_for_dens)
    ax_dens[0].hist(last_xs, bins=n_bins, density = True, label = 'numerical')
    ax_dens[0].plot(x_for_dens, p_x, label = 'exact')
    ax_dens[1].hist(last_ls, bins=n_bins, density = True, label = 'numerical')
    ax_dens[1].plot(l_for_dens, p_l, label = 'exact')
    
    ax_dens[0].set_xlabel("X")
    ax_dens[0].set_ylabel("$p(X)$")
    ax_dens[1].set_xlabel("$\lambda$")
    ax_dens[1].set_ylabel("$p(\lambda)$")
    ax_dens[0].legend()
    ax_dens[1].legend()
    ax_dens[0].set_title("Stocbio - X densities")
    ax_dens[1].set_title("Stocbio - $\lambda$ densities")
    fig_dens.show()
    
    xsm, XCIL, XCIU = get_CI(xsnp)
    lsm, LCIL, LCIU = get_CI(lsnp)

    
    #ax_t.hist(xsnp[:,2000])
    #fig_t.show()
    #print(np.std(xsnp[:,2000]))
    
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
    
    return TS, xs, ls, xsm, XCIL, XCIU, lsm, LCIL, LCIU

    
def plot_simulations_mckean(model,num_sims: int):
    """Plot results of stocbio run alongside confidence intervals"""
    palette_num = get_N_HexCol(num_sims)
    fig, ax = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    #fig3_ci, ax3_ci = plt.subplots(1)
    fig_cloud, ax_cloud = plt.subplots(1)
    fig_xtra, ax_xtra = plt.subplots(1)
    fig_dens, ax_dens = plt.subplots(1,2,tight_layout=True)
    """ Plot several simulations in one image."""
    """Create storage for data over multiple runs"""
    xs_many = []
    ls_many = []
    for i in range(num_sims):
        print("Started iteration " + str(i+1) + " of " + str(num_sims) + ". sigma = " + str(model.SIGMA_X) + ", k_F = " + str(model.F_SCALE))
        TS, xs, ls, x_cloud, extra_term = model.run_mckean()
        xs_many.append(xs)
        ls_many.append(ls)
        cloud_m, cloud_L, cloud_U = get_CI(x_cloud.transpose())
        #cl_many.append(cloud_m)
        #cl_many_L.append(cloud_L)
        #cl_many_U.append(cloud_U)
        ax.plot(TS, ls, palette_num[i])
        ax2.plot(TS, xs, palette_num[i])
        #ax3_ci.plot(TS,cloud_m)
        #ax3_ci.fill_between(TS, cloud_L, cloud_U, color=palette_num[i], alpha=.15)
        ax_cloud.loglog(TS,np.std(x_cloud.transpose(), axis = 0)**2,color=palette_num[i])
        ax_xtra.loglog(TS,np.abs(extra_term),color=palette_num[i])
        
    xsnp = np.asarray(xs_many)
    lsnp = np.asarray(ls_many)
    
    """Fetch the last iterate from every run - needed to compare histograms to expected density."""
    last_xs = xsnp[:,-1]
    last_ls = lsnp[:,-1]
    """Obtain expected densities."""
    #p_x, p_l = compare_distrib(last_xs,last_ls)
    """sort x and lambda with their densities. Otherwise lineplot comes out weird."""
    #xs, p_x_l = zip(*sorted(zip(xs, p_x_l)))
    #ls, p_l = zip(*sorted(zip(ls, p_l)))
    """Define number of bins for histogram"""
    n_bins = 30
    """Retrieve data about min/max/spread of final iterates"""
    min_x = np.min(last_xs)
    max_x = np.max(last_xs)
    xstd = np.std(last_xs)
    min_l = np.min(last_ls)
    max_l = np.max(last_ls)
    lstd = np.std(last_ls)
    """Obtain expected densities."""
    x_for_dens = np.linspace(min_x-xstd,max_x+xstd,1000)
    l_for_dens = np.linspace(min_l-lstd,max_l+lstd,1000)
    p_x = model.get_p_x(x_for_dens)
    p_l = model.get_p_l(l_for_dens)
    ax_dens[0].hist(last_xs, bins=n_bins, density = True, label = 'numerical')
    ax_dens[0].plot(x_for_dens, p_x, label = 'exact')
    ax_dens[1].hist(last_ls, bins=n_bins, density = True, label = 'numerical')
    ax_dens[1].plot(l_for_dens, p_l, label = 'exact')
    
    ax_dens[0].set_xlabel("X")
    ax_dens[0].set_ylabel("$p(X)$")
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
    
    #ax3_ci.set_xlabel("time")
    #ax3_ci.set_ylabel("$X^j$")
    #ax3_ci.set_title("Simulations of Mckean-Vlasov SDE - Cloud C.I. $X$-values")
    #fig3_ci.show()
    
    ax_cloud.set_xlabel("time")
    ax_cloud.set_ylabel("$(\sigma(X^j))^2$")
    ax_cloud.set_title("Simulations of Mckean-Vlasov SDE - Variance of Cloud")
    fig_cloud.show()
    
    ax_xtra.set_xlabel("time")
    ax_xtra.set_ylabel(r"$|\int \nabla_\lambda F \,\mu_t(dx\mid\lambda) - \nabla_{\lambda} F |$")
    ax_xtra.set_title("Simulations of Mckean-Vlasov SDE - Extra term size")
    fig_xtra.show()
    
    """Save plots"""
    dirstr = "sigma_" + str(model.SIGMA_X) + "_k_F_" + str(model.F_SCALE) + "_k_C_" + str(model.C_SCALE) + "_num_sims_" + str(num_sims)
    subfolder_path = os.path.join(sim_path, dirstr)
    iterates_path = os.path.join(subfolder_path, "ALL_ITERATES")
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    if not os.path.exists(iterates_path):
        os.makedirs(iterates_path)
    fig.savefig(iterates_path + '/M_V_lam.png')
    fig2.savefig(iterates_path + '/M_V_x.png')
    #fig3_ci.savefig(subfolder_path + '/M_V_cloud.png')
    fig_dens.savefig(subfolder_path + '/M_V_density.png')
    fig_xtra.savefig(subfolder_path + '/loglog_extra_term.png')
    fig_cloud.savefig(subfolder_path + '/loglog_cloud_var.png')
    
    return TS, xs, ls, xsm, XCIL, XCIU, lsm, LCIL, LCIU


"""CODE BELOW USED FOR RUNNING SIMULATIONS"""
    
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
            model = Model(TOTAL_SIMS, SIGMA, EPS, F_SCALE, C_SCALE, NUM_CLOUD)
            print("Starting Stocbio")
            TS, xs, ls,       xsm1, XCIL1, XCIU1, lsm1, LCIL1, LCIU1 = plot_simulations(model,TOTAL_SIMS)
            print("Starting M-V")
            TS_m, xs_m, ls_m, xsm2, XCIL2, XCIU2, lsm2, LCIL2, LCIU2 = plot_simulations_mckean(model,TOTAL_SIMS)
            
            fig_both_x, ax_x = plt.subplots(1)
            fig_both_l, ax_l = plt.subplots(1)
            
            ax_x.plot(TS, xsm1, label = 'stocbio',color="blue")
            ax_x.fill_between(TS, XCIL1, XCIU1, color="blue", alpha=.15)
            ax_x.plot(TS_m, xsm2, label = 'mckean',color="orange")
            ax_x.fill_between(TS_m, XCIL2, XCIU2, color="orange", alpha=.15)
            ax_x.legend()
            ax_x.set_xlabel("time")
            ax_x.set_ylabel("X")
            str1 = "2-D Problem: " + str(TOTAL_SIMS) + " simulations, " + "+-1 st.d. X-values"
            ax_x.set_title( str1 )
            fig_both_x.show()
            
            ax_l.plot(TS, lsm1, label = 'stocbio',color="blue")
            ax_l.fill_between(TS, LCIL1, LCIU1, color="blue", alpha=.15)
            ax_l.plot(TS_m, lsm2, label = 'mckean',color="orange")
            ax_l.fill_between(TS_m, LCIL2, LCIU2, color="orange", alpha=.15)
            ax_l.legend()
            str2 = "2-D Problem: " + str(TOTAL_SIMS) + " simulations, " + "+-1 st.d. $\lambda$-values"
            ax_l.set_title(str2)
            ax_l.set_xlabel("time")
            ax_l.set_ylabel("$\lambda$")
            fig_both_l.show()
            
            
            dirstr = "sigma_" + str(model.SIGMA_X) + "_k_F_" + str(model.F_SCALE) + "_k_C_" + str(model.C_SCALE) + "_num_sims_" + str(TOTAL_SIMS)
            subfolder_path = os.path.join(sim_path, dirstr)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
            fig_both_x.savefig(subfolder_path + '/pm1std_x.png')
            fig_both_l.savefig(subfolder_path + '/pm1std_lam.png')