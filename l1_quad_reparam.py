# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:17:03 2023

    Simulate langevin for the potential
    U(z,eta) = eps*z^2 / (2*eta) + eta / (2*eps) + G(z)
    
    In our case choose G(t) = (t-2)**2

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

subfolder_name = "SIMULATION"
current_directory = os.getcwd()
sim_path = os.path.join(current_directory, subfolder_name)



class Model_l1_quad:
    """Model contains the necessary parameters/methods to run a single simulation of either Stocbio or M-V sdes."""
    def __init__(self,TOTAL_SIMS,SIGMA_Z,SIGMA_ETA,K_ETA):
        """K_ETA needed to increase sensitivity to eta """
        self.K_ETA = K_ETA
        self.SIGMA_Z = SIGMA_Z
        self.SIGMA_ETA = SIGMA_ETA
        """Runtime-specific parameters"""
        self.T_END = 12
        self.GRID = 4800
        """Epsilon serves to differentiate between the two timesteps of the inner and outer problem"""
        
    def G_z(self,t):
        gxl = (1/2)*(t-2)**2
        return gxl
    
    def grad_G(self,t):
        gxl = (t-2)
        return gxl

    def getU(self,z,eta):
        """Return potential"""
        k_e = self.K_ETA
        return k_e * z**2 / (2*eta) + eta / (2*k_e) +  self.G_z(z)

    def get_gradU_z(self,z,eta):
        k_e = self.K_ETA
        return k_e * z / eta + self.grad_G(z)
        
    def get_gradU_e(self,z,eta):
        k_e = self.K_ETA
        return -k_e * z**2 / (2*eta**2) + 1 / (2*k_e)

    def mu_z(self,z,eta) -> float:
        """Implement the drift term for x (inner, quick sde)."""
        muz = -self.get_gradU_z(z,eta)
        return muz

    def mu_eta(self,z,eta) -> float:
        mueta = -self.get_gradU_e(z,eta)
        return mueta


    def dW(self,delta_t: float) -> float:
        """Sample a random number at each call."""
        return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))


    def run_sde(self):
        """ Return the result of one full simulation of Stocbio."""
        T_INIT = 0
        T_END = self.T_END
        N = self.GRID  # Compute at 1000 grid points
        DT = float(T_END - T_INIT) / N
        TS = np.arange(T_INIT, T_END + DT, DT)

        SIGMA_Z = self.SIGMA_Z
        SIGMA_ETA = self.SIGMA_ETA
        
        Z_INIT = 0
        E_INIT = 1

        zs = np.zeros(TS.size)
        zs[0] = Z_INIT
        es = np.zeros(TS.size)
        es[0] = E_INIT
        for i in range(1, TS.size):
            t = T_INIT + (i - 1) * DT
            z = zs[i - 1]
            e = es[i - 1]
            zs[i] = z + self.mu_z(z, e) * DT + SIGMA_Z * self.dW(DT)
            es[i] = e + self.mu_eta(z, e) * DT + SIGMA_ETA * self.dW(DT)
            """In case this run is divergent, break and return NaN."""
            if np.abs(zs[i]) > 1e5 or np.abs(es[i]) > 1e5:
                return math.nan, math.nan, math.nan

        return TS, zs, es


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
        print("Started iteration " + str(i+1) + " of " + str(num_sims) + ". sigma_Z = " + str(model.SIGMA_Z) + ",sigma_eta = " + str(model.SIGMA_ETA))
        TS, xs, ls = model.run_sde()
        if np.isnan(TS).any() == False:
            xs_many.append(xs)
            ls_many.append(ls)
            ax.plot(TS, ls, palette_num[i])
            ax2.plot(TS, xs, palette_num[i])
        
        
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

    """Obtain expected densities."""

    ax_dens[0].hist(last_xs, bins=n_bins, density = True, label = 'numerical')
    ax_dens[1].hist(last_ls, bins=n_bins, density = True, label = 'numerical')
    
    ax_dens[0].set_xlabel("z")
    ax_dens[0].set_ylabel("$p(z)$")
    ax_dens[1].set_xlabel("$\eta$")
    ax_dens[1].set_ylabel("$p(\eta)$")
    ax_dens[0].legend()
    ax_dens[1].legend()
    ax_dens[0].set_title("z densities")
    ax_dens[1].set_title("$\eta$ densities")
    fig_dens.show()
    
    

    
    #ax_t.hist(xsnp[:,2000])
    #fig_t.show()
    #print(np.std(xsnp[:,2000]))
    
    ax.set_xlabel("time")
    ax.set_ylabel("$\eta$")
    ax.legend()
    ax.set_title("Langevin Simulations- $\eta$-values")
    fig.show()
    
    
    ax2.set_xlabel("time")
    ax2.set_ylabel("$z$")
    ax2.legend()
    ax2.set_title("Langevin Simulations - $Z$-values")
    fig2.show()
    
    """Save plots"""
    dirstr = "sigma_Z" + str(model.SIGMA_Z) + "sigma_eta" + str(model.SIGMA_ETA) + "_num_sims_" + str(num_sims)
    subfolder_path = os.path.join(sim_path, dirstr)
    iterates_path = os.path.join(subfolder_path, "ALL_ITERATES")
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    if not os.path.exists(iterates_path):
        os.makedirs(iterates_path)
    fig.savefig(iterates_path + '/brownian_eta.png')
    fig2.savefig(iterates_path + '/brownian_z.png')
    fig_dens.savefig(subfolder_path + '/brownian_density.png')
    
    #pickled_data = pkl.dumps([TS, xsnp, lsnp])  # returns data as a bytes object
    #compressed_pickle = blosc.compress(pickled_data)
    
    file_path = subfolder_path + '/data_brow.pkl'
    with open(file_path, 'wb') as f:
        pkl.dump([TS, xsnp, lsnp],f)
        #f.write(compressed_pickle)
        
        
    
    return TS, xsnp, lsnp

    
"""CODE BELOW USED FOR RUNNING SIMULATIONS"""
    
if __name__ == "__main__":
    TOTAL_SIMS = 1000
    #SIGMA_ARR = [0.1, 0.5, 1]
    K_ETAS = [0.01,0.1,1]
    #EPS = 0.05
    SIGMA = 0.1

    for K_ETA in K_ETAS:
    #for SIGMA in SIGMA_ARR:
        SIGMA_Z = SIGMA
        SIGMA_ETA = SIGMA
        print("Started for sigma = " + str(SIGMA))
        model = Model_l1_quad(TOTAL_SIMS,SIGMA_Z,SIGMA_ETA,K_ETA)
        print("Starting Langevin")
        TS, xs, ls, = plot_simulations(model,TOTAL_SIMS)
        
        xsm1, XCIL1, XCIU1 = get_CI(xs)
        lsm1, LCIL1, LCIU1 = get_CI(ls)
        
        fig_both_x, ax_x = plt.subplots(1)
        fig_both_l, ax_l = plt.subplots(1)
        
        ax_x.plot(TS, xsm1, color="blue")
        ax_x.fill_between(TS, XCIL1, XCIU1, color="blue", alpha=.15)
        ax_x.legend()
        ax_x.set_xlabel("time")
        ax_x.set_ylabel("Z")
        str1 = "2-D Problem: " + str(TOTAL_SIMS) + " simulations, " + "+-1 st.d. Z-values"
        ax_x.set_title( str1 )
        fig_both_x.show()
        
        ax_l.plot(TS, lsm1, color="blue")
        ax_l.fill_between(TS, LCIL1, LCIU1, color="blue", alpha=.15)
        ax_l.legend()
        str2 = "2-D Problem: " + str(TOTAL_SIMS) + " simulations, " + "+-1 st.d. $\eta$-values"
        ax_l.set_title(str2)
        ax_l.set_xlabel("time")
        ax_l.set_ylabel("$\eta$")
        fig_both_l.show()
        
        
        dirstr = "sigma_Z" + str(model.SIGMA_Z) + "sigma_eta" + str(model.SIGMA_ETA) + "_num_sims_" + str(TOTAL_SIMS)
        subfolder_path = os.path.join(sim_path, dirstr)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        fig_both_x.savefig(subfolder_path + '/pm1std_z.png')
        fig_both_l.savefig(subfolder_path + '/pm1std_eta.png')