# Functions for Discontinuous Galerkin Method
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import *
from ReferenceElement import *
import os
import imageio

# Range Kutta 4 
# This is Tstepper for Evolve
# F is the derivative
def RK4_Step(dt, F,u, K, N,t,a,alpha, M_inv, M_inv_S, q,flux_maxwell, radiative = False):
    w1 = F(u            , K, N, t         ,a, alpha, M_inv, M_inv_S, q, radiative,flux_maxwell)
    w2 = F(u + 0.5*dt*w1, K, N, t + 0.5*dt,a, alpha, M_inv, M_inv_S, q, radiative,flux_maxwell)
    w3 = F(u + 0.5*dt*w2, K, N, t + 0.5*dt,a, alpha, M_inv, M_inv_S, q, radiative,flux_maxwell)
    w4 = F(u + dt*w3    , K, N, t + dt    ,a, alpha, M_inv, M_inv_S, q, radiative,flux_maxwell)
    next_u = u + dt/6*(w1+2*w2+2*w3+w4)
    return next_u

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


# Calculate the numerical flux at point x_k between 2 elements D_(k-1) and D_k
# k-th element D_k covers spatial interval [x_k, x_(k+1)]
# This version is currently for periodic boundary conditions

def f_star_at_x_k(u,k,K,N,t,a,alpha): 
    u_braces = (u[(k-1)%K][N] + u[(k)%K][0])/2                  #average 
    u_brackets = u[(k-1)%K][N] - u[(k)%K][0]                    #difference
    f_star = a*u_braces + np.abs(a)*(1-alpha)/2*u_brackets
 
    return f_star 

# Calculate time derivative for each element D_k
# M_inv_S is the 3rd output of ReferenceElement.py --> ReferenceElement(N)[2], after being scaled
#   by multiplying with 2/h, where 2 come from reference interval interval [-1,1], h is the real width of each element)

def flux_maxwell_E(E,k,K,N,t,a,alpha,H):
    #E and H have similar roles
    E_braces = (E[(k-1)%K][N] + E[(k)%K][0])/2
    H_brackets = H[(k-1)%K][N] - H[(k)%K][0]
    flux = E_braces + (1/a)/2*H_brackets
    if k == 0:
        flux = 0
    return flux 
def flux_maxwell_H(H,k,K,N,t,a,alpha,E):
    #E and H have similar roles
    H_braces = (H[(k-1)%K][N] + H[(k)%K][0])/2
    E_brackets = E[(k-1)%K][N] - E[(k)%K][0]
    flux = H_braces + (1/a)/2*E_brackets
    return flux   

def du_dt_element_k(u,k, K, N,t,a,alpha, M_inv, M_inv_S,q, radiative,flux_maxwell):
    first_term = -np.sqrt(a)*np.matmul(M_inv_S , q[k])
    second_term = +M_inv[:,N] * (a*q[k%K][N]  - flux_maxwell(q,(k+1)%K,K,N,t,a,alpha,u))   #information from element on the right
    third_term  =  -M_inv[:,0] * (a*q[k%K][0]  - flux_maxwell(q,(k)%K  ,K,N,t,a,alpha,u))   #information from element on the left
    du_dt_element = first_term + second_term + third_term
                    
    if radiative == True:
        if k == 0:
            du_dt_element = first_term + second_term
        if k == K-1:
            du_dt_element = first_term + third_term
    return du_dt_element

def DG_du_dt(u, K, N,t,a,alpha, M_inv, M_inv_S,q, radiative,flux_maxwell):
    du_dt_elements = np.empty_like(u)
    for k in range(K):
        du_dt_elements[k] = du_dt_element_k(u,k, K, N,t,a,alpha, M_inv, M_inv_S,q, radiative,flux_maxwell)
    return du_dt_elements


def q_element_k(u,k,K, N,t,a, alpha,M_inv,M_inv_S, radiative):
    first_term = np.sqrt(a)*np.matmul(M_inv_S,u[k])
    second_term = -M_inv[:,N] * (a*u[k%K][N]  - f_star_at_x_k(u,(k+1)%K,K,N,t,a,alpha))
    third_term = M_inv[:,0] * (a*u[k%K][0]  - f_star_at_x_k(u,(k)%K  ,K,N,t,a,alpha))
    q_element = first_term + second_term + third_term
    if radiative == True: 
        if k == 0:
            q_element = first_term + second_term
        if k == K-1 :
            q_element = first_term + third_term
    return q_element

def DG_q_elements(u, K, N,t,a,alpha, M_inv, M_inv_S, radiative = False):
    q_elements = np.empty_like(u)
    for k in range(K):
        q_elements[k] = q_element_k(u,k,K,N,t,a,alpha,M_inv,M_inv_S, radiative)
    return q_elements

