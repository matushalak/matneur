# adex_cython.pyx
import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound

ctypedef np.float64_t DTYPE_t

@boundscheck(False)
@wraparound(False)
cdef double interp_linear(double t,
                          double* t_ptr,
                          double* i_ptr,
                          Py_ssize_t n):
    cdef Py_ssize_t lo = 0, hi = n - 1, mid
    # clamp
    if t <= t_ptr[0]:
        return i_ptr[0]
    elif t >= t_ptr[hi]:
        return i_ptr[hi]
    # binary search for the first index where t_ptr[mid] > t
    while lo < hi:
        mid = (lo + hi) // 2
        if t_ptr[mid] <= t:
            lo = mid + 1
        else:
            hi = mid
    # now lo is the upper bracket
    cdef Py_ssize_t j = lo - 1
    cdef double t0 = t_ptr[j]
    cdef double t1 = t_ptr[j+1]
    cdef double i0 = i_ptr[j]
    cdef double i1 = i_ptr[j+1]
    return i0 + (i1 - i0)*(t - t0)/(t1 - t0)

def adExcython_wrapper(double t,
                    double[:] y,       # state vector
                    dict params,
                    double[:] t_array,
                    double[:] i_array):
    """
    t       : current time
    y       : [V, w]
    t_array : input times
    i_array : input current values
    """
    cdef double Iapp
    cdef double* t_ptr = &t_array[0]
    cdef double* i_ptr = &i_array[0]
    cdef Py_ssize_t   n    = t_array.shape[0]

    # 1) interpolate Iapp at this t
    Iapp = interp_linear(t, t_ptr, i_ptr, n)

    # Extract params (still has dict lookup overhead)
    cdef double C = params['C']
    cdef double gL = params['gL'] 
    cdef double EL = params['EL'] 
    cdef double DeltaT = params['DeltaT'] 
    cdef double VT = params['VT']
    cdef double a = params['a']
    cdef double tauw = params['tauw']

    cdef double V = y[0]
    cdef double w = y[1]

    cdef double dVdt, dwdt # C doubles for output

    # membrane equation
    dVdt = ( -gL*(V - EL)
            + gL*DeltaT*np.exp((V - VT)/DeltaT)
            - w
            + Iapp
            ) / C

    # adaptation variable
    dwdt = ( a*(V - EL) - w )/tauw

    return [dVdt, dwdt]