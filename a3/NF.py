import os
import numpy as np
import scipy.integrate as sp_int
import scipy.optimize as sp_opt
import matplotlib.pyplot as plt
from collections import defaultdict


class CoupledOscillators:
    epsilon_grid = np.linspace(-5, 5, 1000)
    a_grid1 = np.linspace(-1, -0.05, 9)
    tau_grid1 = np.linspace(-.8,.8,1000)
    
    a_grid2 = np.linspace(-1, 0, 1000)
    tau_grid2 = np.linspace(0,1,9) # symmetric around 0
    
    def __init__(self):
        # Sweep a, epsilon and tau grids to find and classify fixed points
        # self.fixed_points = self.sweep1()

        # # Plot results
        # self.plot1()
        
        # Sweep tau, a grids to find and classify fixed points
        self.fixed_points2 = self.sweep2()    
        self.plot2()

    # ∆1 = ∆2 
    @staticmethod
    def PRC(a:float, phase:float):
        return a*np.sin(2*np.pi*phase)
    
    # simplifying assumption that T0=1
    def M(self,
          tau_n:float, epsilon:float, a:float
          )->float:
        '''
        Phase-difference map M(tau_n) -> tau_n+1
        '''
        prc1 = self.PRC(a=a, phase=tau_n)
        # 1 here is T0
        INprc2 = 1 - tau_n - prc1
        prc2 = self.PRC(a=a, phase=INprc2)

        return tau_n + epsilon + prc1 - prc2
    
    
    def sweep1(self)-> dict[int:tuple]:
        ''''
        Performs parameter sweep over alpha, epsilon and tau
        to investigate existence and nature of fixed points
            Looking at epsilon X tau plots for different values of a
        '''
        fixed_points = defaultdict(lambda:dict())

        for ia, a in enumerate(CoupledOscillators.a_grid1):
            for e in CoupledOscillators.epsilon_grid:
                M_vals = self.M(tau_n=CoupledOscillators.tau_grid1,
                                epsilon=e, a = a)
                # At equilibrium, M(tau_n) - tau_n = 0 
                eq_candidates = M_vals - CoupledOscillators.tau_grid1

                equilibria = []
                # Look for sign change to find roots
                for it, (eq_candidate_n, eq_candidate_n1) in enumerate(zip(eq_candidates, eq_candidates[1:])):
                    # sign change
                    if eq_candidate_n * eq_candidate_n1 < 0:
                        try:
                            # accepts functions of 1 variable
                            tau_star = sp_opt.brentq(f = lambda tau: self.M(tau_n=tau,epsilon=e,a = a)-tau,
                                                     a = CoupledOscillators.tau_grid1[it], 
                                                     b = CoupledOscillators.tau_grid1[it+1])
                            # if not already detected for this epsilon
                            if all(abs(tau_star-r)>1e-4 for r in equilibria):
                                equilibria.append(tau_star)
                        except ValueError:
                            pass
                
                # Check stability
                stable:list[bool] = []
                # Synchrony (stable) if derivative of phase difference map < 1
                for t_star in equilibria:
                    dM = (self.M(t_star+1e-6, e, a)-self.M(t_star-1e-6, e, a)) / 2e-6
                    stable.append(abs(dM)<1)
                
                    if abs(dM)<1:
                        print(f'a: {a}, Epsilon: {e}, tau*: {t_star}, Stable: {abs(dM)<1}, dM: {dM}')
                
                if len(equilibria) >= 1:
                    fixed_points[f'{a}'][f'{e}'] = ([e]*len(equilibria), equilibria, stable)
                
                else:
                    print(f'a: {a}, Epsilon: {e}, No equilibria')
        
        return fixed_points
    
    def plot1(self):
        '''
        Looking at epsilon X tau plots for different values of a
        '''
        f, ax = plt.subplots(nrows=3, ncols=3, sharey = 'row', figsize = (9,9))
        # a list of markers for the different plots
        markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']
        for ia, (a_val, a_results) in enumerate(self.fixed_points.items()):
            # choose marker based on index
            marker = markers[ia % len(markers)]

            # get all stable vs unstable epsilons and taus
            all_UNSTABLEeps, all_UNSTABLEtau = [], []
            all_STABLEeps, all_STABLEtau = [], []
            for _, (eps_list, tau_list, stable_list) in a_results.items():
                assert len(stable_list) != 0
                
                stable = np.array(stable_list)
                eps = np.array(eps_list)
                taus = np.array(tau_list)
                
                all_STABLEeps.append(eps[stable])
                all_STABLEtau.append(taus[stable])
                all_UNSTABLEeps.append(eps[~stable])
                all_UNSTABLEtau.append(taus[~stable])

            all_UNSTABLEeps, all_UNSTABLEtau = np.concat(all_UNSTABLEeps), np.concat(all_UNSTABLEtau)
            all_STABLEeps, all_STABLEtau = np.concat(all_STABLEeps), np.concat(all_STABLEtau )
            
            # one scatter per unstable equilibria at a
            ax.flatten()[ia].scatter(all_UNSTABLEeps, all_UNSTABLEtau,
                    marker=marker,
                    color = 'red',
                    linestyle = '--'
                    )
            
            # one scatter per stable equilibria at a
            ax.flatten()[ia].scatter(all_STABLEeps, all_STABLEtau,
                    marker=marker,
                    color = 'navy',
                    )
            
            ax.flatten()[ia].set_title(f'a={round(float(a_val), 2)}')
            
            ax.flatten()[ia].set_xlabel('ε')
            if ia % 3 == 0:
                ax.flatten()[ia].set_ylabel('τ*')
        
        f.tight_layout()
        f.savefig('Phase_Difference_Maps_plots.png', dpi = 300)
        plt.show()


    def sweep_and_plot2(self)-> dict[int:tuple]:
        ''''
        Looking at epsilon X a plots for different values of tau
        '''
        fixed_points = defaultdict(lambda:dict())

        for it, tau in enumerate(CoupledOscillators.tau_grid2):
            for ia, a in enumerate(CoupledOscillators.a_grid2):
            # for e in CoupledOscillators.epsilon_grid:
                M_vals = self.M(tau_n=tau,
                                epsilon=CoupledOscillators.epsilon_grid, a = a)
                # At equilibrium, M(tau_n) - tau_n = 0 
                eq_candidates = M_vals - tau

                equilibriaT, equilibriaE = [], []
                # Look for sign change to find roots
                for ie, (eq_candidate_n, eq_candidate_n1) in enumerate(zip(eq_candidates, eq_candidates[1:])):
                    # sign change
                    if eq_candidate_n * eq_candidate_n1 < 0:
                        try:
                            # accepts functions of 1 variable
                            tau_star = sp_opt.brentq(f = lambda t: self.M(tau_n=t,epsilon=CoupledOscillators.epsilon_grid[ie],a = a)-t,
                                                     a = M_vals[ie], 
                                                     b = M_vals[ie+1])
                            # if not already detected for this epsilon
                            if all(abs(tau_star-r)>1e-4 for r in equilibriaT):
                                equilibriaT.append(tau_star)
                                equilibriaE.append(CoupledOscillators.epsilon_grid[ie])
                        except ValueError:
                            pass
                
                # Check stability
                stable:list[bool] = []
                # Synchrony (stable) if derivative of phase difference map < 1
                for t_star, e_star in zip(equilibriaT, equilibriaE):
                    dM = (self.M(t_star+1e-6, e_star, a)-self.M(t_star-1e-6, e_star, a)) / 2e-6
                    stable.append(abs(dM)<1)
                
                    if abs(dM)<1:
                        print(f'Tau: {tau}, a: {a}, Epsilon: {e_star}, tau*: {t_star}, Stable: {abs(dM)<1}, dM: {dM}')
                
                if len(equilibriaT) >= 1:
                    fixed_points[f'{tau}'][f'{a}'] = ([a]*len(equilibriaE), equilibriaE, stable)
                
                else:
                    print(f'Tau: {tau}, a: {a}, No equilibria')
        
        return fixed_points

    def plot2(self):
        '''
        Looking at epsilon X a plots for different values of tau
        '''
        f, ax = plt.subplots(nrows=3, ncols=3, sharey = 'row', figsize = (9,9))
        # a list of markers for the different plots
        markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']
        for it, (t_val, t_results) in enumerate(self.fixed_points2.items()):
            # choose marker based on index
            marker = markers[it % len(markers)]

            # get all stable vs unstable epsilons and taus
            all_UNSTABLEeps, all_UNSTABLEa = [], []
            all_STABLEeps, all_STABLEa = [], []
            for _, (a_list, eps_list, stable_list) in t_results.items():
                assert len(stable_list) != 0
                
                stable = np.array(stable_list)
                eps = np.array(eps_list)
                ass = np.array(a_list)
                
                all_STABLEeps.append(eps[stable])
                all_STABLEa.append(ass[stable])
                all_UNSTABLEeps.append(eps[~stable])
                all_UNSTABLEa.append(ass[~stable])

            all_UNSTABLEeps, all_UNSTABLEa = np.concat(all_UNSTABLEeps), np.concat(all_UNSTABLEa)
            all_STABLEeps, all_STABLEa = np.concat(all_STABLEeps), np.concat(all_STABLEa)
            
            # one scatter per unstable equilibria at a
            ax.flatten()[it].scatter(all_UNSTABLEeps, all_UNSTABLEa,
                    marker=marker,
                    color = 'red',
                    linestyle = '--'
                    )
            
            # one scatter per stable equilibria at a
            ax.flatten()[it].scatter(all_STABLEeps, all_STABLEa,
                    marker=marker,
                    color = 'navy',
                    )
            
            ax.flatten()[it].set_title(f'τ={round(float(t_val), 2)}')
            
            ax.flatten()[it].set_xlabel('ε')
            if it % 3 == 0:
                ax.flatten()[it].set_ylabel('a')
        
        f.tight_layout()
        f.savefig('a_vs_eps_plots.png', dpi = 300)
        plt.show()




class NeuralFields:
    def __init__(self):
        # Plot kernel and activation
        self.plot_kernel_and_activation()
        

    
    @staticmethod
    def kernel(z:float)->float:
        return (1-np.abs(z))*np.exp(-np.abs(z))
    
    @staticmethod
    def activation(u:float, h:float
                   )->int:
        if isinstance(u, float):
            return 1 if u >= h else 0
        elif isinstance(u, np.ndarray):
            return (u >= h)
    
    def plot_kernel_and_activation(self, 
                                   hplot:float = 0.5):
        # plot synaptic kernel
        fk, kax = plt.subplots()
        kernel_plot_range = (-10,10,100)
        kax.plot(np.linspace(*kernel_plot_range), self.kernel(np.linspace(*kernel_plot_range)),
                 color = 'g')
        kax.set_xlabel('z'); kax.set_ylabel('w(z)')
        fk.tight_layout()
        plt.show()

        # plot Heavyside
        fheavy, heavyax = plt.subplots()
        heavyside_plot_range = (-2,2,100)
        heavyax.plot(np.linspace(*heavyside_plot_range), 
                     self.activation(np.linspace(*heavyside_plot_range), h = hplot),
                     color = 'orange', label = f'h = {round(hplot, 2)}')
        heavyax.set_xlabel('u'); kax.set_ylabel('H(u)')
        heavyax.legend(loc=2)
        fheavy.tight_layout()
        plt.show()

        

if __name__ == '__main__':
    # Question 1 - Coupled Oscillators and Phase Difference Maps
    CO = CoupledOscillators()

    # Questions 2 & 3 - Neural Field Models
    # NF = NeuralFields()
