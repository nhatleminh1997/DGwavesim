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


def numerical_flux_go_with_p(p,q, k, K, N, t, radiative,x_p_dot):
    g = x_p_dot(t) + 1
    h = x_p_dot(t) - 1
    flux = g*h**2*q[k%K][0] - h*g**2*q[(k-1)%K][N] - h**2*p[(k-1)%K][N] + g**2*p[(k)%K][0]     #    WORKING VERSION

    # if radiative == True:
    #     if k == 0:
    #         flux = h**2*q[0][0] - h**2*g*q[0][0] - g**2*q[0][0] + h*g**2*q[0][0]
    #     if k == K:
    #         flux = h**2*q[-1][N]- h**2*g*q[-1][N] - g**2*q[-1][N] + h*g**2*q[-1][N]
    flux =  flux/2
    # if radiative == True:
    #     if k == 0:
    #         flux = p[0][0]
    #     if k == K:
    #         flux = -p[-1][-1]
    return flux

def numerical_flux_go_with_q(p,q, k, K, N, t, radiative,x_p_dot):
    g = x_p_dot(t) + 1
    h = x_p_dot(t) - 1
    flux = g*p[(k-1)%K][N] - h*p[k%K][0] - g*h*q[k%K][0]  + g*h*q[(k-1)%K][N]                #      WORKING VERSION
    if radiative == True:
        if k == 0:
                flux = -p[0][0]
        if k == K:
                flux = -p[-1][-1]
    flux = flux/2

    return flux


def dq_dt_element_k(p,q,k,K,N,t,M_inv,M_inv_S,radiative,delta_source_1,delta_source_2,k_xp, x_p,x_p_dot):
    main_d_dx   = np.matmul(M_inv_S,p[k])
    right_flux  =-M_inv[:,N] * (p[k][N]  - numerical_flux_go_with_q(p,q,k+1,K,N,t,radiative,x_p_dot)) #flux on the right at x_{k+1}
    left_flux   =+M_inv[:,0] * (p[k][0]  - numerical_flux_go_with_q(p,q,k  ,K,N,t,radiative,x_p_dot)) #flux on the left  at x_{k}
    dp_dx_element = main_d_dx + left_flux + right_flux
    # if k == k_xp:
    #     dp_dx_element += (delta_source_1(t)+delta_source_2(t))*M_inv[:,N]/2
    # if k == k_xp+1:
    #     dp_dx_element += -(delta_source_1(t)-delta_source_2(t))*M_inv[:,0]/2
 
    return dp_dx_element

def dq_dt(q,u,p,K, N,t,M_inv,M_inv_S,radiative,delta_source_1,delta_source_2,k_xp,potential,x_p, x_p_dot, x_p_dot_dot):
    dp_dx = np.empty_like(p)
    for k in range(K):
        dp_dx[k] = dq_dt_element_k(p,q,k,K,N,t,M_inv,M_inv_S, radiative,delta_source_1,delta_source_2,k_xp, x_p,x_p_dot)
    return dp_dx
def dp_dt_element_k(p,q,k,K,N,t,M_inv,M_inv_S,radiative,delta_source_1,delta_source_2,k_xp, x_p,x_p_dot):
    dq_dx_term   = np.matmul(M_inv_S,q[k])*(1-x_p_dot(t)**2)
    dp_dx_term   = np.matmul(M_inv_S,p[k])*2*x_p_dot(t)

    right_flux  =-M_inv[:,N] * (2*x_p_dot(t)*p[k][N] + (1 - x_p_dot(t)**2)*q[k][N]  - numerical_flux_go_with_p(p,q,k+1,K,N,t,radiative,x_p_dot)) # at x_{k+1}
    left_flux   =+M_inv[:,0] * (2*x_p_dot(t)*p[k][0] + (1 - x_p_dot(t)**2)*q[k][0]  - numerical_flux_go_with_p(p,q,k  ,K,N,t,radiative,x_p_dot)) # at x_{k}

    dp_dt_element = dq_dx_term + left_flux + right_flux + dp_dx_term 
    # if k == k_xp:
    #     dp_dt_element +=  (delta_source_1(t)+delta_source_2(t))*M_inv[:,N]/2

    # if k == k_xp+1:
    #     dp_dt_element +=  (delta_source_1(t)-delta_source_2(t))*M_inv[:,0]/2

    return dp_dt_element

def dp_dt(p,u,q,K, N,t,M_inv,M_inv_S,radiative,delta_source_1, delta_source_2,k_xp,potential,x_p, x_p_dot, x_p_dot_dot):
    dp_dt_ = np.empty_like(q)
    for k in range(K):
        dp_dt_[k] = dp_dt_element_k(p,q,k,K,N,t,M_inv,M_inv_S, radiative,delta_source_1, delta_source_2,k_xp,x_p,x_p_dot)
    
    try:
        dp_dt_ += -u*potential + x_p_dot_dot(t)*q
    except TypeError:
        print("some error")
        pass
    return dp_dt_

def du_dt(u,p,q,K,N,t,M_inv,M_inv_S,radiative,delta_source_1, delta_source_2, k_xp,potential,x_p, x_p_dot, x_p_dot_dot):
    return p


def RK4_Step(dt, F,u,p,q, K, N,t, M_inv, M_inv_S, radiative = False,delta_source_1 = False, delta_source_2 = False, k_xp = False, potential = False, x_p = False, x_p_dot = False, x_p_dot_dot = False):
    w1 = F(u            ,p,q, K, N, t         , M_inv, M_inv_S, radiative,delta_source_1, delta_source_2,k_xp,potential, x_p, x_p_dot, x_p_dot_dot)
    w2 = F(u + 0.5*dt*w1,p,q, K, N, t + 0.5*dt, M_inv, M_inv_S, radiative,delta_source_1, delta_source_2,k_xp,potential, x_p, x_p_dot, x_p_dot_dot)
    w3 = F(u + 0.5*dt*w2,p,q, K, N, t + 0.5*dt, M_inv, M_inv_S, radiative,delta_source_1, delta_source_2,k_xp,potential, x_p, x_p_dot, x_p_dot_dot)
    w4 = F(u + dt*w3    ,p,q, K, N, t + dt    , M_inv, M_inv_S, radiative,delta_source_1, delta_source_2,k_xp,potential, x_p, x_p_dot, x_p_dot_dot)
    next_u = u + dt/6*(w1+2*w2+2*w3+w4)
    return next_u