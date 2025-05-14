import os
import numpy as np
import scipy.integrate as sp_int
import scipy.optimize as sp_opt
import matplotlib.pyplot as plt
from collections import defaultdict


class CoupledOscillators:
    a_grid1 = np.linspace(-1, -0.05, 9)
    epsilon_grid = np.linspace(-5, 5, 1000)
    tau_grid1 = np.linspace(-.8,.8,1000)
    
    a_grid2 = np.linspace(-1, 0, 1000)
    tau_grid2 = np.linspace(-.8,.8,9) # symmetric around 0
    
    def __init__(self, 
                 n_epsilon:int|None = None,
                 n_tau:int|None = None):
        
        if n_epsilon is not None:
            CoupledOscillators.epsilon_grid = np.linspace(-5, 5, n_epsilon)
        
        if n_tau is not None:
            CoupledOscillators.tau_grid1 = np.linspace(-.8,.8,n_tau)

        # Sweep a, epsilon and tau grids to find and classify fixed points
        self.fixed_points = self.sweep1()

        # Plot results
        self.plot1()
        
        # Investigating coupling strength(a) as a function of period difference e
        self.analytical_plot2()

    # ∆1 = ∆2 
    @staticmethod
    def PRC(a:float, phase:float):
        return a*np.sin(2*np.pi*phase)
    
    # simplifying assumption that T0=1
    def M(self,
          tau_n:float, epsilon:float, a:float
          )->float:
        '''
        Firing_time map M(tau_n) -> tau_n+1
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
                # Synchrony (stable) if derivative of firing time map < 1
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
        f.savefig('Firing_Time_Maps_plots.png', dpi = 300)
        plt.show()

    def analytical_plot2(self)-> dict[int:tuple]:
        ''''
        Analytically at epsilon X a plots for different values of tau
        Since: 
            M(tau, eps, a) = tau + eps + PRC(tau, a) - PRC(1-tau-PRC(tau, a), a)
            At equilibrium M(tau, eps, a) = tau*
            
            tau & tau* cancel out and we get:
                eps* = PRC(1-tau-PRC(tau, a), a) - PRC(tau, a)
                based on derivative at each eps*, a combination, 
                we can determine stability
        '''
        fixed_points = defaultdict(lambda:dict())
        eps_star = lambda tau, a: (self.PRC(phase = 1-tau-self.PRC(phase=tau, a=a), a=a)
                                   - self.PRC(phase=tau, a=a))
        
        tau_results = []
        for it, tau in enumerate(CoupledOscillators.tau_grid2):
                eps_stars = eps_star(tau, a = CoupledOscillators.a_grid2)
                if tau == 0:
                    eps_stars = np.zeros_like(eps_stars)

                # Check stability
                stable:list[bool] = []
                # Synchrony (stable) if derivative of firing time map < 1
                for e_star, a_star in zip(eps_stars, CoupledOscillators.a_grid2):
                    dM = (self.M(tau+1e-6, e_star, a_star)-self.M(tau-1e-6, e_star, a_star)) / 2e-6
                    stable.append(abs(dM)<1)
                
                    if abs(dM)<1:
                        print(f'Tau: {tau}, a: {a_star}, Epsilon: {e_star}, Stable: {abs(dM)<1}, dM: {dM}')
                
                tau_results.append((tau, eps_stars, stable))
        
        # plot
        f, ax = plt.subplots(nrows=3, ncols=3, sharey = 'row', figsize = (9,9))
        # a list of markers for the different plots
        markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']
        for it, (tau, eps_stars, stable_list) in enumerate(tau_results):
            # choose marker based on index
            marker = markers[it % len(markers)]

            # get all stable vs unstable epsilons and taus
            assert len(stable_list) != 0
                
            stable = np.array(stable_list)
            a_stars = CoupledOscillators.a_grid2
            
            all_STABLEeps = eps_stars[stable]
            all_UNSTABLEeps = eps_stars[~stable]
            all_STABLEas = a_stars[stable]
            all_UNSTABLEas = a_stars[~stable]
            
            # one scatter per unstable equilibria at a
            ax.flatten()[it].scatter(all_UNSTABLEeps, all_UNSTABLEas,
                    marker=marker,
                    color = 'red',
                    linestyle = '--'
                    )
            
            # one scatter per stable equilibria at a
            ax.flatten()[it].scatter(all_STABLEeps, all_STABLEas,
                    marker=marker,
                    color = 'navy',
                    )
            
            ax.flatten()[it].set_title(f'τ={round(float(tau), 2)}')
            
            ax.flatten()[it].set_xlabel('ε')
            if it % 3 == 0:
                ax.flatten()[it].set_ylabel('a')
        
        f.tight_layout()
        f.savefig('a_vs_eps_plots.png', dpi = 300)
        plt.show()
        


class NeuralFields:
    # Question 2
    def Amari(self):
        # # Plot kernel
        self.plot_kernel()
        # Plot heavyside
        self.plot_activation()
        
        # # # Plot arbitrary bump with x1 = 0 and x2 = 1
        self.plot_bump(xrange=np.linspace(-10, 10, 200000),
                       x1 = 0, x2 = 1)
        
        # # Plot bump bifurcation diagram
        self.plot_bump_bifurcation(bump_widths=np.linspace(0.0001, 7, 10000),
                                   stability=False)

        # TODO: plot a range of bumps together with their eigenmodes
        # Plot eigenmodes V+(x) and V-(x)
        self.plot_bump(xrange=np.linspace(-10, 15, 200000),
                       x1 = 0, x2 = 6)
        # translation invariant, shift everything to x1 = 0
        x2s = 6*np.ones(200000)#np.linspace(0.0001, 7, 10000)
        x1s = np.zeros_like(x2s)
        x_range = np.linspace(-10, 15, x1s.size)
        self.plot_eigenmodes(C = 1, x_range=x_range, x1=x1s, x2=x2s)

        # Plot full bump bifurcation diagram (with stability)
        self.plot_bump_bifurcation(bump_widths=np.linspace(0.0001, 7, 10000),
                                   stability=True)
        
        # plot sigmoidal firing rate function
        self.plot_activation(type='sigmoidal')
    
    # Question 3
    def Ring(self):
        self.ring_neural_field(kappa = 1, h = .2, time = 20)

    
    @staticmethod
    def kernel(z:float)->float:
        return (1-np.abs(z))*np.exp(-np.abs(z))
    
    @staticmethod
    def activation(u:float, h:float = 1,
                   type:str = 'heavyside'
                   )->int:
        match type:
            case 'heavyside':
                if isinstance(u, float):
                    return 1 if u >= h else 0
                elif isinstance(u, np.ndarray):
                    return (u >= h)
            case 'sigmoidal':
                # sigmoid with µ = 10
                return 1 / (1 + np.exp(-10*(u-h)))
    
    @staticmethod
    def U(x:float, 
          x1:float, x2:float
          )->float:
        # left half of bump + right half of bump
        return (NeuralFields.phi(x-x1)) + (NeuralFields.phi(x2-x))
    
    @classmethod
    def phi(cls,
            z:float
            )-> float:
        return z*np.exp(-np.abs(z))

    # Question 3
    def ring_neural_field(self,
                          kappa: float,
                          h: float,
                          time:int):
        # Spatial grid
        nx = 2000
        Lx = 20
        x = np.linspace(-Lx/2, Lx/2, nx)
        dx = x[1] - x[0]

        # Build circulant weight matrix
        W = np.zeros((nx, nx))
        base = self.kernel(x)
        for i in range(nx):
            W[i, :] = np.roll(base, -i)
        W *= dx

        # RHS for solve_ivp: corrected matrix multiply
        rhs = lambda t, u: -u + kappa*(W @ self.activation(u, h=h, type='sigmoidal'))


        # Prepare 3×3 figure
        # if too narrow start seeing funny things *Turing instabilities and bump fragmenting
        # alphas = np.linspace(0.1, 1.125, 9) # at h = 0.1, kappa = 0.1
        alphas = np.linspace(0.2, 3, 7)
        alphas = np.append(alphas, [20, 50])
        # plot initial conditions
        # Initial conditions
        u0plot = lambda width: (1 / (np.cosh(width*x)**2))
        fini, axini = plt.subplots()
        for ini_param in alphas:
            axini.plot(x, u0plot(ini_param), label = f'α= {round(ini_param, 1)}')
        
        axini.legend(loc = 1)
        axini.set_ylabel('Cortical activation u(x)')
        axini.set_xlabel('Cortical location (x)')
        fini.tight_layout()
        fini.savefig('RING_NF_Initial_conditions.png', dpi = 300)
        plt.show()
        plt.close()
        fig, axes = plt.subplots(3, 3, figsize=(11, 10),
                                 sharex=True, sharey=True)
        for i, (ax, alpha) in enumerate(zip(axes.ravel(), alphas)):
            # initial bump
            u0 = 1/np.cosh(0.5*alpha*x)**2

            # integrate
            sol = sp_int.solve_ivp(rhs, [0, time], u0)
            X, T = np.meshgrid(x, sol.t)
            U = sol.y

            # plot
            cf = ax.contourf(X, T, U.T, levels=15, cmap='viridis')
            ax.set_title(f'α={alpha:.2f}')

            if i % 3 == 0:
                ax.set_ylabel('Time (s)')
            if i>= 6:
                ax.set_xlabel('Cortical location (x)')
        fig.colorbar(cf, ax =axes, orientation= 'vertical', label= 'u(x,t)', 
                     fraction=0.02,pad=-0.1,
                     )
        fig.tight_layout(rect=[0, 0, 0.95, 1.0])
        fig.savefig(f'Ring_NeuralField(kappa{kappa}_h{h}).png', dpi = 300)
        plt.show()


    # Plotting functions
    def plot_eigenmodes(self,
                        C:float, x_range:np.ndarray,
                        x1:float, x2:float):
        assert (x2 > x1).all()
        # V+(x) == U'(x)
        Vplus = C*(self.kernel(x_range - x2) + self.kernel(x_range - x1))
        Vminus = C*(self.kernel(x_range - x2) - self.kernel(x_range - x1))
        # formula for Uprime I derived in 2.3
        Uprime = self.kernel(x_range - x1) + self.kernel(x2-x_range)

        fe, axe = plt.subplots()
        axe.axvline(x = x1[0], label = 'x1', 
                    alpha = 0.6, color = 'k', linestyle = '--')
        axe.axvline(x = x2[0], label = 'x2', 
                    alpha = 0.6, color = 'k', linestyle = '-.')
        
        axe.plot(x_range, Vplus, color = 'green', label = 'V+(x)',
                 linewidth = 2)
        axe.plot(x_range, Vminus, color = 'red', label = 'V-(x)',
                 alpha = 0.7)
        axe.plot(x_range, Uprime,label = "U'(x)", 
                 color = 'blue', linestyle = ':', linewidth = 3)

        axe.set_ylabel('Eigenmode V±(x)')
        axe.set_xlabel('Cortex location x')
        axe.legend(loc=1)
        fe.tight_layout()
        fe.savefig(f'Eigenmodes_∆{np.mean(x2)}.png', dpi = 300)
        plt.show()

    def plot_bump_bifurcation(self, 
                              bump_widths:np.ndarray,
                              h_starval:float = 1/np.exp(1),
                              stability:bool = False):
        h_s = self.phi(bump_widths)
        h_star_i = np.argmin(h_starval - h_s)
        fb, axb = plt.subplots()
        
        # Q2.4
        print(f'Narrow: {h_s[bump_widths<1].size}, Wide: {h_s[bump_widths>1].size}')
        axb.scatter(h_s[bump_widths<1], bump_widths[bump_widths<1], 
                    color = 'pink', label = 'narrow bumps')
        axb.scatter(h_s[bump_widths>1], bump_widths[bump_widths>1], 
                    color = 'yellowgreen', label = 'wide bumps')
        axb.set_xlabel('Firing threshold (h)')
        axb.set_ylabel('Bump width (∆)')
        
        if stability:
            lambda_plus = ((self.kernel(0) + self.kernel(bump_widths)) / 
                           np.abs(self.kernel(0) - self.kernel(bump_widths))
                           ) - 1
            stable = lambda_plus < 0
            unstable = lambda_plus > 0 

            axb.plot(h_s[stable], bump_widths[stable], 
                    color = 'dodgerblue', label = 'stable',
                    linewidth = 3)
            axb.plot(h_s[unstable], bump_widths[unstable], 
                    color = 'red', label = 'unstable', linestyle = '--')

        
        axb.scatter(h_s[h_star_i], bump_widths[h_star_i], 
                    color = 'gold', label = r'${h* = \frac{1}{e}}$', s = 100)
        axb.legend(loc = 1)
        fb.tight_layout()
        fb.savefig(f'Bumb bifurcation_stability({stability}).png', dpi = 300)
        plt.show()

    def plot_bump(self, xrange:np.ndarray,
                  x1:float, x2:float,
                  ax = None):
        # get U steady state values across x
        Ux = self.U(xrange, x1 = x1, x2 = x2)

        if ax is None:
            f, ax = plt.subplots()
            # x < 0
            ax.plot(xrange[xrange<=x1], Ux[xrange<=x1], label= r'$x<x_1$',
                    color = 'silver', linestyle = '--')
            # x1 <= x <= x2
            ax.plot(xrange[((xrange>x1) & (xrange<x2))], Ux[((xrange>x1) & (xrange<x2))], 
                    label = r'${x_1 \leq x \leq x_2}$',
                    color = 'dodgerblue')
            # x > x2
            ax.plot(xrange[xrange>=x2], Ux[xrange>=x2], 
                    label= r'$x > x_2$ ',
                    color = 'black', linestyle = '--')

            ax.axhline(y = self.phi(x2-x1), color = 'orange', label = 'Threshold h')

            ax.set_xlabel('Cortex location x')
            ax.set_ylabel('Cortical activity u(x)')
            ax.legend(loc = 1)

            f.tight_layout()
            f.savefig(f'Bump_plot_∆={np.mean(x2-x1)}.png', dpi = 300)
            plt.show()

        else:
            ax.plot(xrange, Ux)
        
    def plot_kernel(self): 
        # plot synaptic kernel
        fk, kax = plt.subplots()
        kernel_plot_range = (-10,10,1000)
        kax.plot(np.linspace(*kernel_plot_range), self.kernel(np.linspace(*kernel_plot_range)),
                 color = 'g')
        kax.set_xlabel('z'); kax.set_ylabel('w(z)')
        fk.tight_layout()
        plt.show()

    def plot_activation(self,
                        hplot:float = 0.5, 
                        type:str='heavyside'):
        # plot Heavyside
        fheavy, heavyax = plt.subplots()
        heavyside_plot_range = (-2,2,100)
        heavyax.plot(np.linspace(*heavyside_plot_range), 
                     self.activation(np.linspace(*heavyside_plot_range), h = hplot,
                                     type=type),
                     color = 'orange', label = f'h = {round(hplot, 2)}')
        heavyax.set_xlabel('u'); heavyax.set_ylabel('H(u)')
        heavyax.legend(loc=2)
        fheavy.tight_layout()
        plt.show()

        

if __name__ == '__main__':
    # Question 1 - Coupled Oscillators and Phase Difference Maps
    # for truly nice graphs with all the saddle node bifurcations
    # need to set higher resolution for tau and epsilon (both to 10000)
    # at None does default with 1000 each
    CO = CoupledOscillators(n_tau=None, n_epsilon=None)

    # Questions 2 & 3 - Neural Field Models
    NF = NeuralFields()
    
    # Question 2 - Amari approach
    NF.Amari()

    # Question 3 - Ring neural field
    NF.Ring()
