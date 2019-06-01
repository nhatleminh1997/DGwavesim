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
def RK4_Step(dt, F,u, K, N,t,a,alpha, M_inv, M_inv_S, g = None):
    w1 = F(u            , K, N, t         ,a, alpha, M_inv, M_inv_S, g)
    w2 = F(u + 0.5*dt*w1, K, N, t + 0.5*dt,a, alpha, M_inv, M_inv_S, g)
    w3 = F(u + 0.5*dt*w2, K, N, t + 0.5*dt,a, alpha, M_inv, M_inv_S, g)
    w4 = F(u + dt*w3    , K, N, t + dt    ,a, alpha, M_inv, M_inv_S, g)
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
    #if k==0:
    #    f_star = g(t)
        
    return f_star 

# Calculate time derivative for each element D_k
# M_inv_S is the 3rd output of ReferenceElement.py --> ReferenceElement(N)[2], after being scaled
#   by multiplying with 2/h, where 2 come from reference interval interval [-1,1], h is the real width of each element)


def du_dt_element_k(u,k, K, N,t,a,alpha, M_inv, M_inv_S,g=None):
    first_term = -a*np.matmul( M_inv_S , u[k])
    second_term = M_inv[:,N] * (a*u[k%K][N]  - f_star_at_x_k(u,(k+1)%K,K,N,t,a,alpha))   #information from element on the right
    third_term = -M_inv[:,0] * (a*u[k%K][0]  - f_star_at_x_k(u,(k)%K  ,K,N,t,a,alpha))   #information from element on the left
                    
    """if k == K-1:                             #radiative boundary conditions?
        du_dt = first_term + third_term
    else:
        du_dt = first_term + second_term + third_term  """       
    du_dt_element = first_term + second_term + third_term
    return du_dt_element

def DG_du_dt(u, K, N,t,a,alpha, M_inv, M_inv_S,g=None):
    du_dt_elements = np.empty_like(u)
    for k in range(K):
        du_dt_elements[k] = du_dt_element_k(u,k, K, N,t,a,alpha, M_inv, M_inv_S)
    return du_dt_elements


def Evolve(t_initial, t_final, Tstepper, F,CF, start, end, initial_value_function, K, N,a,alpha,g= None):
    h = (end-start)/K

    reference_element = ReferenceElement(N)
    reference_interval = reference_element[0]
    M_inv = reference_element[1]*2/h
    M_inv_S = reference_element[2]*2/h

    x = get_x_elements(start,end, K, reference_interval)
    u = initial_value_function(x,t_initial)

    dx_min = get_dx_min(x)
    dt = CF*dx_min
    nt = int((t_final - t_initial)/dt)          #number of time steps to be evolved 

    t = t_initial

    for n in range(nt):

        u = RK4_Step(dt,F,u, K,N,t,a,alpha,M_inv, M_inv_S,g)
        t = t + dt
    
    return t, u, x
        
# Radiative boundary conditions
def f_star_at_x_k_radiative(u,k,K,N,t,a,alpha): 
    u_braces = (u[(k-1)%K][N] + u[(k)%K][0])/2                  #average 
    u_brackets = u[(k-1)%K][N] - u[(k)%K][0]                    #difference
    f_star = a*u_braces + np.abs(a)*(1-alpha)/2*u_brackets
    if a > 0:
        if k==0:
            f_star = 0
    else: 
        if k == K:
            f_star = 0
    
        
    return f_star 

def du_dt_element_k_radiative(u,k, K, N,t,a,alpha, M_inv, M_inv_S, g = None):
    first_term = -a*np.matmul( M_inv_S , u[k])
    second_term = M_inv[:,N] * (a*u[k][N]  - f_star_at_x_k_radiative(u,k+1,K,N,t,a,alpha))   #information from element on the right
    third_term = -M_inv[:,0] * (a*u[k][0]  - f_star_at_x_k_radiative(u,k  ,K,N,t,a,alpha))   #information from element on the left
    
    du_dt_element = first_term + second_term + third_term
    
    if a > 0: 
        if k == K-1:
            du_dt_element = first_term + third_term
    else:
        if k == 0:
            du_dt_element = first_term + second_term 
    
    #Implememnting delta source g(t) at x = 0 for a>0

    if k ==0:
        du_dt_element += g(t)*M_inv[:,0]


    return du_dt_element

def DG_du_dt_radiative(u, K, N,t,a,alpha, M_inv, M_inv_S,g = None):
    du_dt_elements = np.empty_like(u)
    for k in range(K):
        du_dt_elements[k] = du_dt_element_k_radiative(u,k, K, N,t,a,alpha, M_inv, M_inv_S,g)
    return du_dt_elements


def interpolated_plot(u_elements,x_elements, nx_element):
    interpolated_u = np.empty((len(u_elements), nx_element))
    smooth_x = np.empty_like(interpolated_u)

    for i in range(len(u_elements)):
        smooth_x_element = np.linspace(x_elements[i][0],x_elements[i][-1],nx_element,True)
        interpolated_u_element = lagrange(x_elements[i],u_elements[i])(smooth_x_element)

        smooth_x[i] = smooth_x_element
        interpolated_u[i] = interpolated_u_element
        
        #plotting
        plt.plot(smooth_x_element,interpolated_u_element)    # interpolated lagrange polynomials
        plt.scatter(x_elements[i],u_elements[i])           # nodal points
    