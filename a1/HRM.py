import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy import arange, roots, linspace, meshgrid, array, sign, zeros_like, sqrt
from scipy.integrate import solve_ivp

def phase_portrait(voltage_min:float = -2.1, voltage_max:float = 1.9, voltage_step:float = 0.01,
                   trajectories:bool | list[list[float], list[float]] = array([])):
    fv = lambda v : -v**3 + 3*v**2 
    gv = lambda v : 5*v**2 - 1

    dvdt = lambda v, r, I = 0 : -v**3 + 3*v**2 - r + I
    drdt = lambda v, r : 5*v**2 - 1 - r

    voltages = arange(voltage_min, voltage_max + voltage_step, voltage_step)
    # vectorized
    r_nullcline = gv(voltages)
    v_nullcline = fv(voltages)
    fming = fv(voltages) - gv(voltages)

    # Assemble vector field
    n_vectors = 50
    xv = linspace(min(voltages), max(voltages), n_vectors)
    yr = linspace(min([*r_nullcline,*v_nullcline]), 
                    max([*r_nullcline,*v_nullcline]), n_vectors)

    # all of these are 100 x 100 arrays
    XV, YR = meshgrid(xv, yr)
    DVDT = array([[dvdt(v = x, r = y) for x, y in zip(rows, cols)] for rows, cols in zip(XV, YR)])
    DRDT = array([[drdt(v = x, r = y) for x, y in zip(rows, cols)] for rows, cols in zip(XV, YR)])

    quadrant = (sign(DVDT) > 0).astype(int) + 2 * (sign(DRDT) > 0).astype(int)
    custom_cmap = ListedColormap(['lightcoral', 'blue', 'gold', 'limegreen'])

    # PLOT
    fig, ax = plt.subplots()
    # Equilibria
    equilibria_v = roots([-1, -2, 0, 1]) # what a cool function!
    equilibria_r = gv(equilibria_v)
    eq_labs = ['E1', 'E2', 'E3']

    ax.plot(voltages, v_nullcline, label = 'v nullcline', color = 'deeppink')
    ax.plot(voltages, r_nullcline, label = 'r nullcline', color =  'saddlebrown')
    # ax.plot(voltages, fming, color = 'green') # f(v) - g(v)
    for i, lab in enumerate(eq_labs):
        ax.annotate(lab, 
            (equilibria_v[i], equilibria_r[i]),
            fontsize = 12, fontweight = 'bold',
            bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha = 0.5)
            )
    ax.scatter(equilibria_v, equilibria_r, color = 'k', label = 'Equilibria of HRM')

    if trajectories.size > 0:
        ax.plot(trajectories[0,:], trajectories[1,:], alpha = 0.7, color = 'blue', zorder = 1)


        # gymnastics to get rid of arrow tails
        u = v = 2
        length = sqrt(u**2 + v**2)
        width=0.008
        hal = hl = 1.0 / width * length

        # normalize vectors to have constant size
        U, V = trajectories[2,:], trajectories[3,:]
        U = U / sqrt(U**2 + V**2)
        V = V / sqrt(U**2 + V**2)

        mask = zeros_like(U)
        mask[::1] = 1
        ax.quiver(trajectories[0,:] * mask,
                  trajectories[1,:] * mask,
                  U * mask,
                  V * mask,
                  pivot='tail', angles='xy', scale_units='xy', scale=15,
                  headaxislength=hal, headlength=hl,headwidth=hl, width = width,
                  zorder = 2, color = 'blue')
        
        ax.quiver(XV, YR, DVDT, DRDT, quadrant, 
            cmap = custom_cmap, 
            pivot = 'tip',angles='xy',width=0.002, zorder = 0, alpha = 0.6)
        
        plt.xlim([trajectories[0,:].min() - 0.5, trajectories[0,:].max() + 0.5])
        plt.ylim([trajectories[1,:].min() - 0.5, trajectories[1,:].max() + 0.5])

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
    title =  'trajectories.png' if trajectories.size > 0 else f'HRM phase plane_{round(min(voltages), 2)}:{round(max(voltages), 2)}.png'
    plt.savefig(title, dpi = 200)
    plt.show()
    plt.close()


def jacobian(v:float = False):
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
    
    tr_roots = roots([-3, 6, -1])
    det_roots = roots([3, 4, 0])
    tr_r_lab = [r'$1 + \sqrt{2/3}$', r'$1 - \sqrt{2/3}$']
    det_r_lab = ['-4/3', '0']

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

def HRM(t:float, vars:list[float, float], I:callable):
    '''
    t : float, time, necessary for scipy.solve_ivp
    vars : list[float, float], [v(t), r(t)], [voltage, recovery variable]
    I : float, input current I(t)
    '''
    dvdt = lambda v, r, I = 0 : -v**3 + 3*v**2 - r + I(t)
    drdt = lambda v, r : 5*v**2 - 1 - r
    return array([dvdt(*vars, I),
                  drdt(*vars)])

def voltage_trace(duration: int, v_start:float, r_start:float):
    # Question 6 & 7
    IApplied = lambda t: 0

    # Question 8
    # IApplied = lambda t: 20 if t > 300 and t < 350 else 0

    hrm = lambda t,vars: HRM(t, vars, IApplied)

    t_interval = [0, duration]
    solution = solve_ivp(fun = hrm, t_span = t_interval, t_eval= linspace(0,duration, 2000), 
                         y0 = [v_start, r_start])

    ts = solution.t
    
    voltage, recovery  = solution.y
    applied_current = [IApplied(tm) for tm in ts]


    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(ts, voltage, color = 'k')
    ax.set_ylabel('Membrane Voltage (v)')
    
    # ax[1].plot(ts, recovery, color = 'lightsteelblue')
    # ax[1].set_ylabel('Recovery (r)')

    plt.xlabel('Time (t)')
    plt.savefig('voltage_trace.png', dpi = 200)
    plt.show()
    plt.close()
    
    return array([list(voltage[:-1]), 
                  list(recovery[:-1]),
                 [(voltage[i+1] - voltage[i]) / (ts[i+1] - ts[i]) for i in range(voltage.size-1)], #dvdt
                 [(recovery[i+1] - recovery[i]) / (ts[i+1] - ts[i]) for i in range(voltage.size-1)]]) #drdt

if __name__ == '__main__':
    # phase_portrait() 
    # jacobian()
    # nature_equilibria()
    
    # with all of these, also include trajectory along phase portrait
    # Q6 : self-sustained spikes
    # sustained_traj = voltage_trace(500, .5, 0)
    # phase_portrait(trajectories=sustained_traj)

    # Q7 : Bistability
    # e3
    # e3_traj = voltage_trace(500, -0.99, 2)
    # phase_portrait(trajectories=e3_traj)

    # e1
    e1_traj = voltage_trace(500, -1.5, 5)
    phase_portrait(trajectories=e1_traj)

    # Q8 : Suppressing periodic spikes w current
    # suppress = voltage_trace(500, -0.99, 2)
    # phase_portrait(trajectories=suppress, voltage_max=3.5)