import numpy as np

def d_dx_periodic(u,a,dx):
    du = np.zeros(len(u))
    if a > 0:
        du[1:] = u[1:] - u[:-1]
        du[0] = u[0] - u[-1]
    if a < 0:
        du[:-1] = u[:-1]-u[1:]
        du[-1] = u[-1] - u[0]
        du = -du
    return du/dx

def d_dt(u,a,dx,dt,du_dx, v):
    try:
        d_dt = -a*du_dx(u,a,dx) + v
    except TypeError:
        d_dt = -a*du_dx(u,a,dx) 
    return d_dt


def du_dx_radiative(u,a,dx):
    du = np.zeros(len(u))
    if a > 0:
        du[1:] = u[1:] - u[:-1]
        du[0] = u[0]
    if a < 0:
        du[:-1] = u[:-1]-u[1:]
        du[-1] = u[-1]
        du = -du
    return du/dx

def RK4_step(u,a, dx,dt,du_dx,du_dt,v = None):
    w1 = du_dt(u,            a,dx,dt,du_dx, v)
    w2 = du_dt(u + 0.5*dt*w1,a,dx,dt,du_dx, v)
    w3 = du_dt(u + 0.5*dt*w2,a,dx,dt,du_dx, v)
    w4 = du_dt(u + dt*w3    ,a,dx,dt,du_dx, v)
    return u + dt*(w1 + 2.0*w2 + 2.0*w3 + w4)/6.0

