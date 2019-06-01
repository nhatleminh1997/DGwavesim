#fd_functions_heat_equation
import numpy as np
#central finite differencing 1st order

def d_dx_periodic(u,a,dx):
    du = np.zeros(len(u))
    
    du[1:-1] = u[2:] - u[:-2]
    du[0]    = u[1]  - u[-1]
    du[-1]   = u[0]  - u[-2]

    return np.sqrt(a)*du/(2*dx)

def RK4_step(u,a, dx,dt,du_dt,q): 
    w1 = du_dt(u,            a,dx,dt,q)
    w2 = du_dt(u + 0.5*dt*w1,a,dx,dt,q)
    w3 = du_dt(u + 0.5*dt*w2,a,dx,dt,q)
    w4 = du_dt(u + dt*w3    ,a,dx,dt,q)
    return u + dt*(w1 + 2.0*w2 + 2.0*w3 + w4)/6.0

def d_dt(u,a, dx,dt,q):
    
    dq_dx = d_dx_periodic(q,a,dx)
    return np.sqrt(a)*dq_dx
