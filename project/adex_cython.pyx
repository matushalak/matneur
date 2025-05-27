from libc.math cimport exp
cimport numpy as np
import numpy as np

# This C function remains the same
cdef int calculate_derivatives_c(double t, double V, double w,
                                double C,
                                double gL, double EL, double DeltaT, double VT, 
                                double a, double tauw, double I,
                                double* dVdt_ptr, double* dwdt_ptr):
    dVdt_ptr[0] = (-gL * (V - EL) + gL * DeltaT * exp((V - VT)/DeltaT) - w + I) / C
    dwdt_ptr[0] = (a * (V - EL) - w) / tauw
    return 0

# Python wrapper function - THIS is what solve_ivp calls.
# It MUST match the signature expected by solve_ivp + args.
# Note: We pass 'params' and 'Iapp' via the 'args' tuple.
def adExModel_cython_wrapper(double t, np.ndarray[double, ndim=1] y, dict params, double Ival):
    # cdef double V = y[0] # Using np.ndarray is better
    # cdef double w = y[1]
    cdef double V = y[0]
    cdef double w = y[1]
    cdef double dVdt, dwdt # C doubles for output

    # Extract params (still has dict lookup overhead)
    cdef double C = params['C']
    cdef double gL = params['gL'] 
    cdef double EL = params['EL'] 
    cdef double DeltaT = params['DeltaT'] 
    cdef double VT = params['VT']
    cdef double a = params['a']
    cdef double tauw = params['tauw']

    # --- CORRECTLY call the C function ---
    calculate_derivatives_c(t, V, w,               # Pass inputs by value
                          C, gL, EL, DeltaT, VT,  # Pass params by value
                          a, tauw, Ival,          # Pass I_val by value
                          &dVdt, &dwdt)          # Pass outputs by address

    # Return as a Python list (or NumPy array)
    return [dVdt, dwdt]