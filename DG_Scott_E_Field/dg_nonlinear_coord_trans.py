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

def numerical_flux_go_with_p(p,q, k, K, t, radiative,beta_k,dx_dxi_k,side,k_p,j1,j2):
    if side == 'left':
        p_braces = (p[k][0]+p[(k-1)%K][-1])/2
        q_braces = (q[k][0]+q[(k-1)%K][-1])/2
        p_brackets = p[(k-1)%K][-1]-p[k][0]
        q_brackets = q[(k-1)%K][-1]-q[k][0]
        if radiative == True and k==0:
            p_braces = p[k][0]
            q_braces = -p[k][0]*dx_dxi_k[0]
            p_brackets = 0
            q_brackets = 0
        flux = -beta_k[0]*p_braces + dx_dxi_k[0]**(-2)*q_braces + 1/(2*dx_dxi_k[0])*(p_brackets-beta_k[0]*q_brackets)
        if k == k_p +1:
            j1_ = j1(beta_k[0],dx_dxi_k[0],t)
            j2_ = j2(beta_k[0],dx_dxi_k[0],t)
            flux += 1/2*(j1_+j2_/dx_dxi_k[0])
            # flux += -1/2*(beta_k[0]-1/dx_dxi_k[0])*(j1_ + j2_/dx_dxi_k[0])

    elif side == 'right':
        p_braces = (p[k%K][-1] + p[(k+1)%K][0])/2
        q_braces = (q[k%K][-1] + q[(k+1)%K][0])/2
        p_brackets = p[k%K][-1] - p[(k+1)%K][0]
        q_brackets = q[k%K][-1] - q[(k+1)%K][0]
        if radiative == True and k==K-1 :
            p_braces = p[k][-1]
            q_braces = p[k][-1]*dx_dxi_k[-1]
            p_brackets = 0
            q_brackets = 0
        flux = -beta_k[-1]*p_braces + dx_dxi_k[-1]**(-2)*q_braces + 1/(2*dx_dxi_k[-1])*(p_brackets-beta_k[-1]*q_brackets) 
        if k == k_p:
            j1_ = j1(beta_k[-1],dx_dxi_k[-1],t)
            j2_ = j2(beta_k[-1],dx_dxi_k[-1],t)
            flux += 1/2*(-j1_+j2_/dx_dxi_k[-1])
            # flux += 1/2*(beta_k[-1]+1/dx_dxi_k[-1])*(j2_/dx_dxi_k[-1]-j1_)
    else:
        raise ValueError('Invalid value for side')
    return flux

def numerical_flux_go_with_q(p,q, k, K, t, radiative,beta_k,dx_dxi_k,side,k_p,j1,j2):
    if side == 'left':
        p_braces = (p[k%K][0]+p[(k-1)%K][-1])/2
        q_braces = (q[k%K][0]+q[(k-1)%K][-1])/2
        p_brackets = p[(k-1)%K][-1]-p[k%K][0]
        q_brackets = q[(k-1)%K][-1]-q[k%K][0]
        if radiative == True and k ==0:
            p_braces = -q[k][0]/dx_dxi_k[0]
            q_braces = q[k][0]
            p_brackets = 0
            q_brackets = 0
        flux = p_braces - beta_k[0]*q_braces + 1/2*(1/dx_dxi_k[0]*q_brackets -dx_dxi_k[0]*beta_k[0]*p_brackets)
        if k == k_p+1:
            j1_ = j1(beta_k[0],dx_dxi_k[0],t)
            j2_ = j2(beta_k[0],dx_dxi_k[0],t)
            flux += 1/2*(dx_dxi_k[0]*j1_+j2_)
            # flux += -1/2*(beta_k[0]-1/dx_dxi_k[0])*(j1_*dx_dxi_k[0]+j2_)

    elif side == 'right':
        p_braces = (p[k%K][-1]+p[(k+1)%K][0])/2
        q_braces = (q[k%K][-1]+q[(k+1)%K][0])/2
        p_brackets = -p[(k+1)%K][0]+p[k%K][-1]
        q_brackets = -q[(k+1)%K][0]+q[k%K][-1]
        if radiative == True and k == K-1:
            p_braces = q[k][-1]/dx_dxi_k[-1] 
            q_braces = q[k][-1]
            p_brackets = 0
            q_brackets = 0
        flux = p_braces - beta_k[-1]*q_braces + 1/2*(1/dx_dxi_k[-1]*q_brackets -dx_dxi_k[-1]*beta_k[-1]*p_brackets)
        if k == k_p:
            j1_ = j1(beta_k[-1],dx_dxi_k[-1],t)
            j2_ = j2(beta_k[-1],dx_dxi_k[-1],t)
            flux += 1/2*(dx_dxi_k[-1]*j1_-j2_)
            # flux += 1/2*(beta_k[-1]+1/dx_dxi_k[-1])*(j1_*dx_dxi_k[-1]-j2_)
    else:
        raise ValueError('Invalid value for side')
    return flux

def dq_dt_element_k(p,q,k,K,t,M_inv,M_inv_S,radiative,j1,j2,k_p,x_p,x_p_dot,beta_k,dx_dxi_k):
    first_term = np.matmul(M_inv_S,q[k])*beta_k 
    second_term = -np.matmul(M_inv_S,p[k])
    left_flux = -M_inv[:, 0]*(p[k][ 0]- beta_k[ 0]*q[k][ 0]-numerical_flux_go_with_q(p,q,k,K,t,radiative,beta_k,dx_dxi_k,'left' ,k_p,j1,j2))
    right_flux=  M_inv[:,-1]*(p[k][-1]- beta_k[-1]*q[k][-1]-numerical_flux_go_with_q(p,q,k,K,t,radiative,beta_k,dx_dxi_k,'right',k_p,j1,j2))
    # if k == k_p+1:
    #     left_flux = -M_inv[:, 0]*(-numerical_flux_go_with_q(p,q,k,K,t,radiative,beta_k,dx_dxi_k,'left' ,k_p,j1,j2))
    # if k == k_p:
    #     right_flux=  M_inv[:,-1]*(-numerical_flux_go_with_q(p,q,k,K,t,radiative,beta_k,dx_dxi_k,'right',k_p,j1,j2))
    dp_dx_element = first_term + second_term + left_flux + right_flux
    return dp_dx_element

def func_dq_dt(q,u,p,K,t,M_inv,M_inv_S,radiative,j1,j2,k_p,potential,x_p, x_p_dot,beta,dx_dxi,d2x_dxi2):
    dq_dt = np.empty_like(p)
    for k in range(K):
        dq_dt[k] = dq_dt_element_k(p,q,k,K,t,M_inv,M_inv_S, radiative,j1,j2,k_p, x_p,x_p_dot,beta[k],dx_dxi[k])
    dq_dt += d2x_dxi2*q 
    return dq_dt

def dp_dt_element_k(p,q,k,K,t,M_inv,M_inv_S,radiative,j1,j2,k_p, x_p,x_p_dot,beta_k,dx_dxi_k):
    first_term = np.matmul(M_inv_S,p[k])*beta_k
    second_term = -1/dx_dxi_k**2*np.matmul(M_inv_S,q[k])
    left_flux = -M_inv[:, 0]*(-beta_k[ 0]*p[k][ 0] + dx_dxi_k[ 0]**(-2)*q[k][ 0]- numerical_flux_go_with_p(p,q, k, K, t, radiative,beta_k,dx_dxi_k,'left' ,k_p,j1,j2))
    right_flux = M_inv[:,-1]*(-beta_k[-1]*p[k][-1] + dx_dxi_k[-1]**(-2)*q[k][-1]- numerical_flux_go_with_p(p,q, k, K, t, radiative,beta_k,dx_dxi_k,'right',k_p,j1,j2))
    # if k == k_p+1:
    #     left_flux = -M_inv[:, 0]*(-numerical_flux_go_with_p(p,q, k, K, t, radiative,beta_k,dx_dxi_k,'left' ,k_p,j1,j2))
    # if k == k_p: 
    #     right_flux = M_inv[:,-1]*(-numerical_flux_go_with_p(p,q, k, K, t, radiative,beta_k,dx_dxi_k,'right',k_p,j1,j2))
    dp_dt_element_k = first_term + second_term + left_flux + right_flux
    return dp_dt_element_k

def func_dp_dt(p,u,q,K,t,M_inv,M_inv_S,radiative,j1,j2,k_p,potential,x_p, x_p_dot,beta,dx_dxi,d2x_dxi2):
    dp_dt = np.empty_like(q)
    for k in range(K):
        dp_dt[k] = dp_dt_element_k(p,q,k,K,t,M_inv,M_inv_S, radiative,j1,j2,k_p,x_p,x_p_dot,beta[k],dx_dxi[k])
    dp_dt += d2x_dxi2/dx_dxi**3*q
    try:
        dp_dt += -potential*u
    except TypeError:
        print('Error in adding potential term for dp_dt')
        pass
    return dp_dt

def func_du_dt(u,p,q,K,t,M_inv,M_inv_S,radiative,j1, j2,k_p,potential,x_p, x_p_dot,beta,dx_dxi,d2x_dxi2):
    return beta*q - p

def RK4_Step(dt, F,u,p,q, K,t, M_inv, M_inv_S, radiative = False,j1 = False, j2 = False, k_p = False, potential = False, x_p = False, x_p_dot = False, beta = False, dx_dxi = False,d2x_dxi2 = False):
    w1 = F(u            ,p,q, K, t         , M_inv, M_inv_S, radiative, j1, j2,k_p,potential, x_p, x_p_dot,beta,dx_dxi,d2x_dxi2)
    w2 = F(u + 0.5*dt*w1,p,q, K, t + 0.5*dt, M_inv, M_inv_S, radiative, j1, j2,k_p,potential, x_p, x_p_dot,beta,dx_dxi,d2x_dxi2)
    w3 = F(u + 0.5*dt*w2,p,q, K, t + 0.5*dt, M_inv, M_inv_S, radiative, j1, j2,k_p,potential, x_p, x_p_dot,beta,dx_dxi,d2x_dxi2)
    w4 = F(u + dt*w3    ,p,q, K, t + dt    , M_inv, M_inv_S, radiative, j1, j2,k_p,potential, x_p, x_p_dot,beta,dx_dxi,d2x_dxi2)
    next_u = u + dt/6*(w1+2*w2+2*w3+w4)
    return next_u

def func_dx_dlambda(a,b,t,xi_p,xi,x_p_dot):
    return (xi-a)*(b-xi)*x_p_dot(t)/((xi_p-a)*(b-xi_p))
def func_dx_dxi(a,b,t,xi_p,xi,x_p):
    return ((2*xi-xi_p-a)*(xi_p-x_p(t))+(x_p(t)-a)*(b-xi_p))/((xi_p-a)*(b-xi_p))
def func_d2x_dxi2(a,b,t,xi_p,x_p):
    return 2*(xi_p-x_p(t))/((xi_p-a)*(b-xi_p))
def func_beta(a,b,t,xi_p,xi,x_p,x_p_dot):
    return func_dx_dlambda(a,b,t,xi_p,xi,x_p_dot)/func_dx_dxi(a,b,t,xi_p,xi,x_p)
def func_A(t, xi_p,x_p):
    return 2*(x_p(t)-xi_p)
def func_B(a,b,t,xi_p,x_p):
    return 2*(a*b+xi_p**2-(a+b)*x_p(t))
def func_C(a,b,t,xi_p,x_p):
    return (a**2+b**2)*x_p(t)-a*(b-xi_p)**2-b*(a**2+xi_p**2)
def func_dbeta_dxi(a,b,t,xi_p,xi,x_p,x_p_dot):
    return (func_A(t,xi_p,x_p)*xi**2+func_B(a,b,t,xi_p,x_p)*xi+func_C(a,b,t,xi_p,x_p))*x_p_dot(t)/((2*xi-xi_p-a)*(xi_p-x_p(t))+(x_p(t)-a)*(b-xi_p))**2