# cython: language_level=3
import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound

ctypedef np.float64_t DTYPE_t

# ---------- Parameters packaged in a C struct ----------
cdef struct Params:
    double C, gL, EL, DeltaT, VT
    double a, tau_w, b
    double Vpeak, Vreset

@boundscheck(False)
@wraparound(False)
cpdef tuple integrate_adex_rk4(
        double[:]   I,             # pre-computed input current, length N
        double      dt,
        dict params,
        double      Tmax,
        double      V0,
        double      w0,
    ):
    """
    Returns number of samples stored in t/V/w and number of spikes in spike_out.
    (We encode the latter by returning it directly; caller slices arrays.)
    """
    cdef Py_ssize_t N = I.shape[0]
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t n_spk = 0

    # Pre-allocate output arrays
    cdef np.ndarray[np.float64_t, ndim=1] t_out   = np.empty(N, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] V_out   = np.empty(N, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] w_out   = np.empty(N, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] spk_out = np.empty(int(Tmax/dt) + 8,dtype=np.float64)

    # Copy params dict into C struct
    cdef Params p
    p.C      = params['C']
    p.gL     = params['gL']
    p.EL     = params['EL']
    p.VT     = params['VT']
    p.DeltaT = params['DeltaT']
    p.Vreset = params['Vreset']
    p.Vpeak  = params['Vpeak']
    p.a      = params['a']
    p.tau_w  = params['tau_w']
    p.b      = params['b']

    cdef double t  = 0.0
    cdef double V  = V0
    cdef double w  = w0
    cdef double V_prev

    cdef double k1_V, k2_V, k3_V, k4_V
    cdef double k1_w, k2_w, k3_w, k4_w
    cdef double I_k

    while t < Tmax and idx < N:
        # store current state
        t_out[idx] = t
        V_out[idx] = V
        w_out[idx] = w

        I_k = I[idx]

        # ---- RK4 ----
        k1_V = (-p.gL*(V - p.EL) + p.gL*p.DeltaT*np.exp((V - p.VT)/p.DeltaT) - w + I_k) / p.C
        k1_w = (p.a*(V - p.EL) - w) / p.tau_w

        k2_V = (-p.gL*((V + 0.5*dt*k1_V) - p.EL)
                + p.gL*p.DeltaT*np.exp(((V + 0.5*dt*k1_V) - p.VT)/p.DeltaT)
                - (w + 0.5*dt*k1_w) + I_k) / p.C
        k2_w = (p.a*((V + 0.5*dt*k1_V) - p.EL) - (w + 0.5*dt*k1_w)) / p.tau_w

        k3_V = (-p.gL*((V + 0.5*dt*k2_V) - p.EL)
                + p.gL*p.DeltaT*np.exp(((V + 0.5*dt*k2_V) - p.VT)/p.DeltaT)
                - (w + 0.5*dt*k2_w) + I_k) / p.C
        k3_w = (p.a*((V + 0.5*dt*k2_V) - p.EL) - (w + 0.5*dt*k2_w)) / p.tau_w

        k4_V = (-p.gL*((V + dt*k3_V) - p.EL)
                + p.gL*p.DeltaT*np.exp(((V + dt*k3_V) - p.VT)/p.DeltaT)
                - (w + dt*k3_w) + I_k) / p.C
        k4_w = (p.a*((V + dt*k3_V) - p.EL) - (w + dt*k3_w)) / p.tau_w

        V_prev = V
        V += dt*(k1_V + 2*k2_V + 2*k3_V + k4_V)/6.0
        w += dt*(k1_w + 2*k2_w + 2*k3_w + k4_w)/6.0

        t  += dt
        idx += 1

        # -------- spike detection --------
        if V_prev < p.Vpeak <= V:          # crossed upward through Vpeak
            # linear interpolation for more accurate spike time
            spk_out[n_spk] = t - dt + dt*(p.Vpeak - V_prev)/(V - V_prev)
            n_spk += 1

            # reset
            V = p.Vreset
            w += p.b

    # once loop is done
    return (t_out[:idx].copy(),
            V_out[:idx].copy(),
            w_out[:idx].copy(),
            spk_out[:n_spk].copy() )