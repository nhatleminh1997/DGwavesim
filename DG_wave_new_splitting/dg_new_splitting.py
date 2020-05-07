# Functions for Discontinuous Galerkin Method
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import *
from ReferenceElement import *
import os
import imageio


# Get optimal LGL spatial grid-points for Discontinuous Galerkin method
# Return an array of K arrays of grid-points, one for each element D_k
# Parameter reference_interval is the first output of ReferenceElement(N), 
# which returns LGL collocation points on reference interval [-1,1] to be mapped to our real interval [start,end]

def get_x_elements(start, end, K, reference_interval): #LGL points
    h = (end-start)/K #Element width
    x_elements = []
    for k in range(K):
        element = []
        for r_i in reference_interval:
            element.append(start + k*h+(r_i+1)/2*h)
        x_elements.append(element)
    return np.asarray(x_elements)

#Get smallest spatial spacing dxmin in a DG scheme with LGL points
#Output used with Courant factor to calculate suitable size for time step dt 

def get_dx_min(x_elements):
    a = x_elements[0]
    dxarray = np.empty_like(a)
    for i in range(len(a)):
        dxarray[i] = np.abs(a[i]-a[(i+1)%len(a)])
    return np.min(dxarray)




def numerical_flux_q(p,q, k, K,N,t,radiative):
    q_braces   = (q[(k-1)%K][N] + q[(k)%K][0])/2                  #average 
    p_brackets =  p[(k-1)%K][N] - p[(k)%K][0]                    #difference
    flux = 0.5*p_brackets - q_braces
    if radiative == True:
        if k == 0:
            flux = -p[0][0]
        if k == K:
            flux = p[-1][-1]
    return -flux

def numerical_flux_p(p,q, k, K,N,t,radiative):
    p_braces   = (p[(k-1)%K][N] + p[(k)%K][0])/2                  #average 
    q_brackets =  q[(k-1)%K][N] - q[(k)%K][0]                    #difference
    flux = 0.5*q_brackets - p_braces
    if radiative == True:
        if k == 0:
            flux = -q[0][0]
        if k == K:
            flux = q[-1][-1]
    return -flux
def dp_dx_element_k(p,q,k,K,N,t,M_inv,M_inv_S,radiative):
    main_d_dx   = np.matmul(M_inv_S,p[k])
    right_flux  =-M_inv[:,N] * (p[k][N]  - numerical_flux_p(p,q,k+1,K,N,t,radiative)) #flux on the right at x_{k+1}
    left_flux   = M_inv[:,0] * (p[k][0]  - numerical_flux_p(p,q,k  ,K,N,t,radiative)) #flux on the left  at x_{k}
    dp_dx_element = main_d_dx + left_flux + right_flux
    """if radiative == True:
        if k == 0:
            dp_dx_element = main_d_dx + right_flux 
        if k == K-1:
            dp_dx_element = main_d_dx + left_flux """
    return dp_dx_element

def dq_dt(q,u,p,K, N,t,M_inv,M_inv_S,radiative):
    dp_dx = np.empty_like(p)
    for k in range(K):
        dp_dx[k] = dp_dx_element_k(p,q,k,K,N,t,M_inv,M_inv_S, radiative)
    return dp_dx
def dq_dx_element_k(p,q,k,K,N,t,M_inv,M_inv_S,radiative):
    main_d_dx   = np.matmul(M_inv_S,q[k])
    right_flux  =-M_inv[:,N] * (q[k][N]  - numerical_flux_q(p,q,k+1,K,N,t,radiative)) # at x_{k+1}
    left_flux   = M_inv[:,0] * (q[k][0]  - numerical_flux_q(p,q,k  ,K,N,t,radiative)) # at x_{k}
    dq_dx_element = main_d_dx + left_flux + right_flux
    """if radiative == True:
        if k == 0:
            dq_dx_element = main_d_dx + right_flux 
        if k == K-1:
            dq_dx_element = main_d_dx + left_flux """
    return dq_dx_element

def dp_dt(p,u,q,K, N,t,M_inv,M_inv_S,radiative):
    dq_dx = np.empty_like(q)
    for k in range(K):
        dq_dx[k] = dq_dx_element_k(p,q,k,K,N,t,M_inv,M_inv_S, radiative)
    return dq_dx
def du_dt(u,p,q,K,N,t,M_inv,M_inv_S,radiative):
    return p


def RK4_Step(dt, F,u,p,q, K, N,t, M_inv, M_inv_S, radiative = False):
    w1 = F(u            ,p,q, K, N, t         , M_inv, M_inv_S, radiative)
    w2 = F(u + 0.5*dt*w1,p,q, K, N, t + 0.5*dt, M_inv, M_inv_S, radiative)
    w3 = F(u + 0.5*dt*w2,p,q, K, N, t + 0.5*dt, M_inv, M_inv_S, radiative)
    w4 = F(u + dt*w3    ,p,q, K, N, t + dt    , M_inv, M_inv_S, radiative)
    next_u = u + dt/6*(w1+2*w2+2*w3+w4)
    return next_u