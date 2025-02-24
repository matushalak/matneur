import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy import arange, roots, linspace, meshgrid, array, sign

step =  0.01
fv = lambda v : -v**3 + 3*v**2 
gv = lambda v : 5*v**2 - 1

dvdt = lambda v, r, I = 0 : -v**3 + 3*v**2 - r + I
drdt = lambda v, r : 5*v**2 - 1 - r

voltages = arange(-2.1, 1.9+step, step)
r_nullcline = [gv(v) for v in voltages]
v_nullcline = [fv(v) for v in voltages]
fming = [fv(v) - gv(v) for v in voltages]

# Assemble vector field
n_vectors = 50
xv = linspace(min(voltages), max(voltages), n_vectors)
yr = linspace(min(r_nullcline+v_nullcline), 
              max(r_nullcline+v_nullcline), n_vectors)

# all of these are 100 x 100 arrays
XV, YR = meshgrid(xv, yr)
DVDT = array([[dvdt(v = x, r = y) for x, y in zip(rows, cols)] for rows, cols in zip(XV, YR)])
DRDT = array([[drdt(v = x, r = y) for x, y in zip(rows, cols)] for rows, cols in zip(XV, YR)])

quadrant = (sign(DVDT) > 0).astype(int) + 2 * (sign(DRDT) > 0).astype(int)
custom_cmap = ListedColormap(['lightcoral', 'blue', 'gold', 'limegreen'])

fig, ax = plt.subplots()
#vector field
ax.quiver(XV, YR, DVDT, DRDT, quadrant, 
          cmap = custom_cmap, pivot = 'tip',
            scale=10,               # adjust scale so vectors aren't too short
            scale_units='xy',       # keep scaling consistent in data units
            angles='xy',            # interpret angles in data coordinates
            width=0.001)#,            # arrow line width
            # headwidth=3,            # bigger arrowhead
            # headlength=5,
            # minshaft=0.5,             # ensure minimum arrow shaft
            # minlength=0.5)           # ensure minimum arrow length)

# Equilibria
equilibria_v = roots([-1, -2, 0, 1]) # what a cool function!
equilibria_r = [gv(ev) for ev in equilibria_v]
eq_labs = ['E1', 'E2', 'E3']

ax.plot(voltages, v_nullcline, label = 'v nullcline', color = 'deeppink')
ax.plot(voltages, r_nullcline, label = 'r nullcline', color =  'saddlebrown')
# ax.plot(voltages, fming, color = 'green') # f(v) - g(v)
for i, lab in enumerate(eq_labs):
    ax.annotate(lab, 
        (equilibria_v[i], equilibria_r[i]),
        fontsize = 12, fontweight = 'bold',
        bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha = 0.5),
        )
ax.scatter(equilibria_v, equilibria_r, color = 'k', label = 'Equilibria of HRM')

ax.set_xlabel('v')
ax.set_ylabel('r')
ax.legend(loc = 1, framealpha = 0.9)
plt.tight_layout()
plt.savefig(f'HRM phase plane_{round(min(voltages), 2)}:{round(max(voltages), 2)}.png', dpi = 200)
plt.show()
