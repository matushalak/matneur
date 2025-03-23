import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy import arange, roots, linspace, meshgrid, array, sign, zeros_like, ndarray, stack, append
from numpy.random import uniform
from scipy.integrate import solve_ivp
import os

def phase_portrait(voltage_min:float = -2.1, voltage_max:float = 1.9, voltage_step:float = 0.01,
                   trajectories:bool | ndarray = array([])):
    '''
    Universal function used a lot to calculate plot the phase portrait with vector field, as well as, nullclines and solution trajectories of HRM
    '''
    # HRM equations
    fv = lambda v : -v**3 + 3*v**2 
    gv = lambda v : 5*v**2 - 1

    dvdt = lambda v, r, I = 0 : -v**3 + 3*v**2 - r + I
    drdt = lambda v, r : 5*v**2 - 1 - r

    # setup
    voltages = arange(voltage_min, voltage_max + voltage_step, voltage_step)
    # vectorized
    r_nullcline = gv(voltages)
    v_nullcline = fv(voltages)
    fming = fv(voltages) - gv(voltages)

    # Assemble vector field
    n_vectors = 100
    xv = linspace(min(voltages)-0.5, max(voltages)+0.5, n_vectors)
    yr = linspace(min([*r_nullcline,*v_nullcline])-3, 
                    max([*r_nullcline,*v_nullcline])+2, n_vectors)

    # all of these are 100 x 100 arrays
    XV, YR = meshgrid(xv, yr)
    DVDT = array([[dvdt(v = x, r = y) for x, y in zip(rows, cols)] for rows, cols in zip(XV, YR)])
    DRDT = array([[drdt(v = x, r = y) for x, y in zip(rows, cols)] for rows, cols in zip(XV, YR)])

    quadrant = (sign(DVDT) > 0).astype(int) + 2 * (sign(DRDT) > 0).astype(int) # colors give interpretability to different areas of phase plane
    custom_cmap = ListedColormap(['lightcoral', 'blue', 'gold', 'limegreen'])

    # PLOT
    fig, ax = plt.subplots()
    # Equilibria
    equilibria_v = roots([-1, -2, 0, 1]) # what a cool function!
    equilibria_r = gv(equilibria_v)
    eq_labs = ['E1', 'E2', 'E3']

    ax.plot(voltages, v_nullcline, label = 'v nullcline', color = 'deeppink', zorder = 0)
    ax.plot(voltages, r_nullcline, label = 'r nullcline', color =  'saddlebrown', zorder = 0)
    # ax.plot(voltages, fming, color = 'green') # f(v) - g(v)
    for i, lab in enumerate(eq_labs):
        ax.annotate(lab, 
            (equilibria_v[i], equilibria_r[i]),
            fontsize = 12, fontweight = 'bold',
            bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha = 0.5)
            )
    ax.scatter(equilibria_v, equilibria_r, color = 'k', label = 'Equilibria of HRM', zorder = 1)

    # SOLUTION trajectories
    if trajectories.size > 0:
        for tr in range(trajectories.shape[1]):
            ax.plot(trajectories[0,tr,:], trajectories[1,tr,:], alpha = 0.9, zorder = 2, linestyle = ':')
            ax.scatter(trajectories[0,tr,0], trajectories[1,tr,0], marker = '*')

        # background vector field
        ax.quiver(XV, YR, DVDT, DRDT, quadrant, 
            cmap = custom_cmap, 
            pivot = 'tip',angles='xy',width=0.0025, zorder = 0, alpha = 0.4)
        
        plt.xlim([trajectories[0,:].min() - 0.5, trajectories[0,:].max() + 0.5])
        plt.ylim([trajectories[1,:].min() - 3, trajectories[1,:].max() + 2])

        ax.legend(loc = 1, framealpha = 0.9)

    else:
        #vector field
        ax.quiver(XV, YR, DVDT, DRDT, quadrant, 
                  cmap = custom_cmap, 
                  pivot = 'tip',angles='xy',width=0.0015)
        ax.legend(loc = 1, framealpha = 0.9)
        

    ax.set_xlabel('v')
    ax.set_ylabel('r')
    
    plt.tight_layout()
    if trajectories.size > 0:
        traj_name = f'trajectories_{sum(file.startswith("trajectories") for file in os.listdir())}.png'
    title =  traj_name if trajectories.size > 0 else f'HRM phase plane_{round(min(voltages), 2)}:{round(max(voltages), 2)}.png'
    plt.savefig(title, dpi = 300)
    plt.show()
    plt.close()


def jacobian(v:float = False):
    '''
    Calculates and plots the roots of the trace and determinant of the Jacobian of HRM
    '''
    if v:
        J = array([-3*(v**2) + 6*v, -1],
                [10*v, -1])
        return J
    
    tr = lambda v:  -3*(v**2) + 6*v - 1
    det = lambda v: 3*(v**2) + 4*v

    voltages = arange(-1.75, 2.1 + 0.01, 0.001)

    # vectorised
    trace = tr(voltages)
    determinant = det(voltages)
    
    # cool function to calculate roots of polynomials!
    tr_roots = roots([-3, 6, -1])
    det_roots = roots([3, 4, 0])
    tr_r_lab = [r'$1 + \sqrt{2/3}$', r'$1 - \sqrt{2/3}$']
    det_r_lab = ['-4/3', '0']

    # plot characterizing the Jacobian of HRM based on sign of trace and determinant
    plt.plot(voltages, trace, label = r'$\tau(v)$')
    plt.plot(voltages, determinant, label = r'$\Delta(v)$')
    plt.hlines(0, min(voltages), max(voltages), colors='k')
    plt.scatter(tr_roots, zeros_like(tr_roots))
    plt.scatter(det_roots, zeros_like(tr_roots))

    for i in range(2):
        plt.annotate(tr_r_lab[i], [tr_roots[i], -5],
                    fontsize = 12, fontweight = 'bold', ha = 'center')
        plt.annotate(det_r_lab[i], [det_roots[i], 3],
                    fontsize = 12, fontweight = 'bold', ha = 'center')
    plt.xlabel('v')
    plt.ylabel(r'$\tau(v)$ or $\Delta(v)$')
    plt.legend()

    plt.tight_layout()
    plt.savefig('TraceDetRoots.png', dpi = 200)
    plt.show()
    plt.close()


def nature_equilibria():
    '''
    Draws the trace-determinant diagram and positions HRM equilibria within it
    '''
    tr = lambda v:  -3*(v**2) + 6*v - 1
    det = lambda v: 3*(v**2) + 4*v

    tr_det = lambda v: tr(v)**2 - 4*det(v)
    voltages = arange(-10, 5 + 0.01, 0.001)
    # roots
    tr_roots = roots([-3, 6, -1])
    det_roots = roots([3, 4, 0])
    
    traces = tr(voltages)
    dets = det(voltages)

    # det is yaxis
    trx = linspace(min(traces), 10, 1000)
    tr_det_parabola = lambda tr: tr**2 / 4

    # EQUILIBRIA
    equilibria_v = roots([-1, -2, 0, 1]) # what a cool function!
    # equilibrium tr, det
    tr_eq = tr(equilibria_v)
    det_eq = det(equilibria_v)
    eq_labs = ['E1', 'E2', 'E3']


    plt.vlines(0, -100, 100, colors='k')
    plt.hlines(0, -100, 100, colors='k') 
    plt.plot(traces, dets, label = r'$\tau(v) \text{ and } \Delta(v)$' + ' across v')
    plt.plot(trx, tr_det_parabola(trx), c = 'green', label= r'$\Delta(v) = \frac{\tau(v)^2}{4}$')
    
    # equilibria
    plt.scatter(tr_eq, det_eq, color = 'red')
    offset_x = [0.3, 0, 0.5]
    offset_y = [0,-3,0]
    for i, lab in enumerate(eq_labs):
        plt.annotate(lab, 
            (tr_eq[i]+offset_x[i], det_eq[i]+offset_y[i]),
            fontsize = 12, fontweight = 'bold',
            )
    
    plt.ylim(min(dets) - 8,  50)
    plt.xlim(-25,  max(traces) + 5)
    plt.ylabel(r'$\Delta(v)$')
    plt.xlabel(r'$\tau(v)$')
    plt.legend(loc = 6)
    plt.tight_layout()
    plt.savefig('Poincare_diagram.png', dpi = 200)
    plt.show()
    plt.close()

def HRM(t:float, vars:list[float, float], I:callable)->ndarray:
    '''
    t : float, time, necessary for scipy.solve_ivp
    vars : list[float, float], [v(t), r(t)], [voltage, recovery variable]
    I : float, input current I(t)
    '''
    dvdt = lambda v, r, I = 0 : -v**3 + 3*v**2 - r + I(t)
    drdt = lambda v, r : 5*v**2 - 1 - r
    return array([dvdt(*vars, I),
                  drdt(*vars)])

def voltage_trace(duration: int, 
                  v_start:float | ndarray, 
                  r_start:float | ndarray,
                  applied_currents:ndarray = array([0]))->ndarray:
    '''
    Given a duration of simulation, set of initial conditions (v, r), and set of applied current amplitudes,
    Calculates, and plots the voltage traces, potentially in reponse to a 15 second burst of applied current of specified strength.

    Returns solution trajectories and their derivatives
    '''
    if type(v_start) == float:
        v_start = [v_start]
        r_start = [r_start]
    
    t_interval = [0, duration]
    TRAJECTORIES = []

    # calculate solutions separately for different starting conditions & applied currents
    for vs, rs in zip(v_start, r_start):
        for IAPP in applied_currents:
            IApplied = lambda t: IAPP if t > 95 and t < 110 else 0
            hrm = lambda t,vars: HRM(t, vars, IApplied)
            solution = solve_ivp(fun = hrm, t_span = t_interval, t_eval= linspace(0,duration, 2000), 
                                y0 = [vs, rs])
            TRAJECTORIES.append(solution.y)

    ts = solution.t
    TRAJECTORIES = array(TRAJECTORIES)

    voltage, recovery  = TRAJECTORIES[:,0,:], TRAJECTORIES[:,1,:]
    applied_current = [IApplied(tm) for tm in ts]

    # plotting traces
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for ind in range(TRAJECTORIES.shape[0]):
        if len(applied_currents) > 1:
            ax.plot(ts, voltage[ind,:], alpha = 0.5, label = f'I_App = {round(applied_currents[ind], 2)}')
        else:
            ax.plot(ts, voltage[ind,:], alpha = 0.5)
    ax.set_ylabel('Membrane Voltage (v)')
    if applied_currents.size > 0:
        plt.legend(loc = 1)

    plt.xlabel('Time (t)')
    plt.tight_layout()
    plt.savefig(f'voltage_trace_{sum(file.startswith("voltage") for file in os.listdir())}.png', dpi = 300)
    plt.show()
    plt.close()

    # derivatives of solution trajectories
    dv_dt, dr_dt = zeros_like(voltage[:,:-1]), zeros_like(voltage[:,:-1])
    for tr in range(voltage.shape[0]):
        for i in range(ts.size-1):
            dv_dt[tr, i] = (voltage[tr,i+1] - voltage[tr,i]) / (ts[i+1] - ts[i])
            dr_dt[tr,i] = (recovery[tr,i+1] - recovery[tr,i]) / (ts[i+1] - ts[i])

    return stack([voltage[:,:-1], 
                  recovery[:,:-1],
                  dv_dt, 
                  dr_dt])

if __name__ == '__main__':
    # ANALYTICAL RESULTS
    # Q2,3
    phase_portrait() 
    # # Q4
    jacobian()
    # # Q5
    nature_equilibria()
    
    # NUMERICAL RESULTS
    # for plotting multiple trajectories around equilibria
    gv = lambda v : 5*v**2 - 1
    equilibria_v = roots([-1, -2, 0, 1]) # what a cool function!
    equilibria_r = gv(equilibria_v)

    aroundEQ = uniform([equilibria_v-0.5, equilibria_r-9], 
                       [equilibria_v+1, equilibria_r+3], 
                       (7,2,3))

    # with all of these, also include trajectory along phase portrait
    # Q6 : self-sustained spikes around E3
    sustained_traj = voltage_trace(150, aroundEQ[:,0,2], aroundEQ[:,1,2])
    phase_portrait(trajectories=sustained_traj)

    # Q7 : Bistability
    # Around E2 Bistability (more widespread sampling)
    aroundEQ2 = uniform([equilibria_v-0.5, equilibria_r-6], 
                    [equilibria_v+1, equilibria_r+8], 
                    (7,2,3))
    e2_traj = voltage_trace(150, aroundEQ2[:,0,1], aroundEQ2[:,1,1])
    phase_portrait(trajectories=e2_traj)

    # # Zooom-in on E1
    e1_traj = voltage_trace(150, aroundEQ[:,0,0], aroundEQ[:,1,0])
    phase_portrait(trajectories=e1_traj)

    # Q8 : Suppressing periodic spikes w current, initial conditions chosen in periodic spiking range
    suppress = voltage_trace(200, -0.99, 2, applied_currents=append(linspace(-10,20, 9), 0))
    phase_portrait(trajectories=suppress, voltage_min=-4, voltage_max=4)