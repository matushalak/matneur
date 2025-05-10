import os
import numpy as np
import scipy.integrate as sp_int
import scipy.optimize as sp_opt
import matplotlib.pyplot as plt
from collections import defaultdict


class CoupledOscillators:
    epsilon_grid = np.linspace(-5, 5, 1000)
    a_grid = np.linspace(-1, -0.05, 9)
    tau_grid = np.linspace(-.8,.8,1000)

    def __init__(self):
        # Sweep a, epsilon and tau grids to find and classify fixed points
        self.fixed_points = self.sweep()

        # Plot results
        self.plot()

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
        INprc2 = 1 - tau_n - prc1
        prc2 = self.PRC(a=a, phase=INprc2)

        return tau_n + epsilon + prc1 - prc2
    
    
    def sweep(self)-> dict[int:tuple]:
        fixed_points = defaultdict(lambda:dict())

        for ia, a in enumerate(CoupledOscillators.a_grid):
            for e in CoupledOscillators.epsilon_grid:
                M_vals = self.M(tau_n=CoupledOscillators.tau_grid,
                                epsilon=e, a = a)
                # At equilibrium, M(tau_n) - tau_n = 0 
                eq_candidates = M_vals - CoupledOscillators.tau_grid

                equilibria = []
                # Look for sign change to find roots
                for it, (eq_candidate_n, eq_candidate_n1) in enumerate(zip(eq_candidates, eq_candidates[1:])):
                    # sign change
                    if eq_candidate_n * eq_candidate_n1 < 0:
                        try:
                            # accepts functions of 1 variable
                            tau_star = sp_opt.brentq(f = lambda tau: self.M(tau_n=tau,epsilon=e,a = a)-tau,
                                                     a = CoupledOscillators.tau_grid[it], 
                                                     b = CoupledOscillators.tau_grid[it+1])
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

    def plot(self):
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
        plt.show()



class NeuralFields:
    def __init__(self):
        pass
        

if __name__ == '__main__':
    # Q1 - Coupled Oscillators
    CO = CoupledOscillators()

    # Questions 2 & 3 - Neural Field Models
    NF = NeuralFields()