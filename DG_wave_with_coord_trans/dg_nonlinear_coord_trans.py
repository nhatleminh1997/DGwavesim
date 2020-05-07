import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import *
from ReferenceElement import *
import os
import imageio

def get_x_elements(start, end, K, reference_interval): #LGL points
    h = (end-start)/K #Element width
    x_elements = []
    for k in range(K):
        element = []
        for r_i in reference_interval:
            element.append(start + k*h+(r_i+1)/2*h)
        x_elements.append(element)
    return np.asarray(x_elements)

def get_dx_min(x_elements):
    a = x_elements[0]
    dxarray = np.empty_like(a)
    for i in range(len(a)):
        dxarray[i] = np.abs(a[i]-a[(i+1)%len(a)])
    return np.min(dxarray)

def numerical_flux_go_with_p(p,q, k, K, N, t, radiative,x_p_dot):
    g = x_p_dot(t) + 1
    h = x_p_dot(t) - 1
    flux = g**2*h*q[k%K][0] - g*h**2*q[(k-1)%K][N] + h**2*p[(k-1)%K][N] - g**2*p[(k)%K][0]   
    flux =  flux/2
    return flux

def numerical_flux_go_with_q(p,q, k, K, N, t, radiative,x_p_dot):
    g = x_p_dot(t) + 1
    h = x_p_dot(t) - 1
    flux = h*p[(k-1)%K][N] - g*p[k%K][0] + g*h*q[k%K][0]  - g*h*q[(k-1)%K][N]                   
    flux = flux/2
    return flux

def dq_dt_element_k(p,q,k,K,N,t,M_inv,M_inv_S,radiative,delta_source_1,delta_source_2,k_xp, x_p,x_p_dot):
    main_d_dx   = np.matmul(M_inv_S,p[k])
    left_flux   = -M_inv[:,0] * (-p[k][0]  - numerical_flux_go_with_q(p,q,k  ,K,N,t,radiative,x_p_dot)) #flux on the left  at x_{k}
    right_flux  = +M_inv[:,N] * (-p[k][N]  - numerical_flux_go_with_q(p,q,k+1,K,N,t,radiative,x_p_dot)) #flux on the right at x_{k+1}
    dp_dx_element = main_d_dx + left_flux + right_flux
    return dp_dx_element

def dq_dt(q,u,p,K, N,t,M_inv,M_inv_S,radiative,delta_source_1,delta_source_2,k_xp,potential,x_p, x_p_dot, x_p_dot_dot):
    dp_dx = np.empty_like(p)
    for k in range(K):
        dp_dx[k] = dq_dt_element_k(p,q,k,K,N,t,M_inv,M_inv_S, radiative,delta_source_1,delta_source_2,k_xp, x_p,x_p_dot)
    return dp_dx

def dp_dt_element_k(p,q,k,K,N,t,M_inv,M_inv_S,radiative,delta_source_1,delta_source_2,k_xp, x_p,x_p_dot):
    dq_dx_term   = np.matmul(M_inv_S,q[k])*(1-x_p_dot(t)**2)
    dp_dx_term   = np.matmul(M_inv_S,p[k])*2*x_p_dot(t)
    left_flux   = -M_inv[:,0] * (-2*x_p_dot(t)*p[k][0] + (-1 + x_p_dot(t)**2)*q[k][0]  - numerical_flux_go_with_p(p,q,k  ,K,N,t,radiative,x_p_dot)) # at x_{k}
    right_flux  = +M_inv[:,N] * (-2*x_p_dot(t)*p[k][N] + (-1 + x_p_dot(t)**2)*q[k][N]  - numerical_flux_go_with_p(p,q,k+1,K,N,t,radiative,x_p_dot)) # at x_{k+1}
    dp_dt_element = dq_dx_term + left_flux + right_flux + dp_dx_term 
    return dp_dt_element

def dp_dt(p,u,q,K, N,t,M_inv,M_inv_S,radiative,delta_source_1, delta_source_2,k_xp,potential,x_p, x_p_dot, x_p_dot_dot):
    dp_dt_ = np.empty_like(q)
    for k in range(K):
        dp_dt_[k] = dp_dt_element_k(p,q,k,K,N,t,M_inv,M_inv_S, radiative,delta_source_1, delta_source_2,k_xp,x_p,x_p_dot)
    try:
        dp_dt_ += x_p_dot_dot(t)*q
    except TypeError:
        print("some error")
        pass
    return dp_dt_

def du_dt(u,p,q,K,N,t,M_inv,M_inv_S,radiative,delta_source_1, delta_source_2, k_xp,potential,x_p, x_p_dot, x_p_dot_dot):
    return p 

def RK4_Step(dt, F,u,p,q, K, N,t, M_inv, M_inv_S, radiative = False,delta_source_1 = False, delta_source_2 = False, 
k_xp = False, potential = False, x_p = False, x_p_dot = False, x_p_dot_dot = False):
    w1 = F(u            ,p,q, K, N, t         , M_inv, M_inv_S, radiative,delta_source_1, delta_source_2,k_xp,potential, x_p, x_p_dot, x_p_dot_dot)
    w2 = F(u + 0.5*dt*w1,p,q, K, N, t + 0.5*dt, M_inv, M_inv_S, radiative,delta_source_1, delta_source_2,k_xp,potential, x_p, x_p_dot, x_p_dot_dot)
    w3 = F(u + 0.5*dt*w2,p,q, K, N, t + 0.5*dt, M_inv, M_inv_S, radiative,delta_source_1, delta_source_2,k_xp,potential, x_p, x_p_dot, x_p_dot_dot)
    w4 = F(u + dt*w3    ,p,q, K, N, t + dt    , M_inv, M_inv_S, radiative,delta_source_1, delta_source_2,k_xp,potential, x_p, x_p_dot, x_p_dot_dot)
    next_u = u + dt/6*(w1+2*w2+2*w3+w4)
    return next_u

def dx_dlambda(a,b,t,xi_p,xi,x_p_dot):
    return (xi-a)*(b-xi)*x_p_dot(t)/((xi_p-a)*(b-xi_p))
def dx_dxi(a,b,t,xi_p,xi,x_p):
    return ((2*xi-xi_p-a)*(xi_p-x_p(t))+(x_p(t)-a)*(b-xi_p))/((xi_p-a)*(b-xi_p))
def d2x_dxi2(a,b,t,xi_p,x_p):
    return 2*(xi_p-x_p(t))/((xi_p-a)*(b-xi_p))
def beta(a,b,t,xi_p,xi):
    return dx_dlambda(a,b,t,xi_p,xi,x_p_dot)/dx_dxi(a,b,t,xi_p,xi,x_p)
def A(t, xi_p,x_p):
    return 2*(x_p(t)-xi_p)
def B(a,b,t,xi_p,x_p):
    return 2*(a*b+xi_p**2-(a+b)*x_p(t))
def C(a,b,t,xi_p,x_p):
    return (a**2+b**2)*x_p(t)-a*(b-xi_p)**2-b*(a**2+xi_p**2)
def dbeta_dxi(a,b,t,xi_p,xi,x_p,x_p_dot):
    return (A(t,xi_p,x_p)*xi**2+B(a,b,t,xi_p,x_p)*xi+C(a,b,t,xi_p,x_p))*x_p_dot(t)/((2*xi-xi_p-a)*(xi_p-x_p(t))+(x_p(t)-a)*(b-xi_p))**2