import numpy as np
from scipy.special import *

def rs_to_ab(r,s):
    # transfer from (r,s) -> (a,b) coordinates in triangle
    Np = len(r)
    a = np.zeros(Np)

    for n in range(len(Np)):
        if s[n] != 1:
            a[n] = 2*(1+r[n]))/(1-s[n])-1
        else: 
            a[n] = -1
    b = s 
    return a,b

def jacobi_gl(N, alpha, beta):
    #compute the nth order Gauss Lobatto quadrature points, r, associated with the Jacobi polynomial, of type (alpha, beta)
    
    if(N==1):
        r=np.array([-1.,1.])
    else:
        inner_roots, inner_weights = roots_jacobi(N-1, alpha+1, beta+1)
        r = np.concatenate([ [-1.], inner_roots, [1.] ])
    return np.transpose(r)

def jacobi_p(x, alpha, beta, n):
    #evaluate jacobi polynomial of type (alpha,beta) > -1 at points x for order N and returns an array 1xlen(x_p)
    
    dims = len(x)
    if dims == 1:
        xp = np.transpose(xp)
    
    PL = np.zeros((N+1, length(xp)))
    
    # initial values P_0(x) and P_1(x)
    gamma0 = 2**(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+1)
    PL[0,:] = 1/np.sqrt(gamma0)
    
    if N == 0:
        P = np.transpose(PL)
        return P
    
    gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
    PL[1,:] = ((alpha+beta+2)*xp/2 + (alpha-beta)/2)/sqrt(gamma1)
    
    if N == 1:
        P = np.transpose(PL(N,:))
        return P
    
    aold  = 2/(2+alpha+beta)*np.sqrt((alpha+1)*(beta+1)/(alpha+beta+3))

    for i in range(1,N):
        h1 = 2*i+alpha+beta
        anew = 2/(h1+2)*np.sqrt( (i+1)*(i+1+alpha+beta)*(i+1+alpha)*(i+1+beta)/(h1+1)/(h1+3))
        bnew = - (alpha**2-beta**2)/h1/(h1+2)
        PL[i+1,:] = 1/anew*(-aold*PL[i-1,:]+(xp-bnew)*PL[i,:])
        aold = anew
    
    P = np.transpose(PL[N,:])

    return P









