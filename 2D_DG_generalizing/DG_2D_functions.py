import numpy as np
from scipy.special import *


def jacobi_gl(alpha, beta, N):
    #compute the nth order Gauss Lobatto quadrature points, r, associated with the Jacobi polynomial, of type (alpha, beta)
    
    if(N==1):
        r=np.array([-1.,1.])
    else:
        inner_roots, inner_weights = roots_jacobi(N-1, alpha+1, beta+1)
        r = np.concatenate([ [-1.], inner_roots, [1.] ])
    return np.transpose(r)

def jacobi_p(x, alpha, beta, N):
    #evaluate jacobi polynomial of type (alpha,beta) > -1 at points x for order N and returns an array 1xlen(x_p)
    xp = x
    dims = len(x)
    if dims == 1:
        xp = np.transpose(xp)
    
    PL = np.zeros((N+1, len(x)))
    
    # initial values P_0(x) and P_1(x)
    gamma0 = 2**(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+1)
    PL[0,:] = 1/np.sqrt(gamma0)
    
    if N == 0:
        P = np.transpose(PL)
        return P
    
    gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
    row2nd = ((alpha+beta+2)*xp/2 + (alpha-beta)/2)/np.sqrt(gamma1)
    for j in range(len(xp)):
        PL[1,j] = row2nd[j] 
    
    if N == 1:
        P = np.transpose(PL[N,:])
        return P
    
    aold  = 2/(2+alpha+beta)*np.sqrt((alpha+1)*(beta+1)/(alpha+beta+3))
    for i in range(1,N):
        h1 = 2*i+alpha+beta
        anew = 2/(h1+2)*np.sqrt( (i+1)*(i+1+alpha+beta)*(i+1+alpha)*(i+1+beta)/(h1+1)/(h1+3))
        bnew = - (alpha**2-beta**2)/h1/(h1+2)
 
        PL[i+1] = 1/anew*(-aold*PL[i-1]+(xp-bnew)*PL[i])
        aold = anew
    
    P = np.transpose(PL[N,:])

    return P

def vandermonde_1d(N,r):
    #initialize the 1D Vandermonde matrix
    V1D = np.zeros((len(r), N+1))
    for j in range(N+1):
        column = jacobi_p(r,0,0,j)
        for i in range(len(r)):
            V1D[i,j] = column[i] 
    return V1D

def warpfactor(N, rout):
    #compute scaled warp function at order N based on rout interpolation nodes
    
    #compute LGL and equidistant node distribution
    LGLr = jacobi_gl(0,0,N)
    req = np.transpose(np.linspace(-1,1,N+1))

    #compute V based on req
    Veq = vandermonde_1d(N, req)

    #evaluate Lagrange polynomial at rout
    Nr = len(rout)
    Pmat = np.zeros((N+1,Nr))
    for i in range(N+1):
        row = jacobi_p(rout,0,0, i)
        for j in range(Nr):
            Pmat[i,j] = row[j] 
    
    Lmat = np.matmul(np.linalg.inv(np.transpose(Veq)), Pmat)
    #compute warp factor
    warp = np.matmul(np.transpose(Lmat),LGLr - req)

    #scale factor
    zerof = np.zeros(len(rout))
    for i in range(len(zerof)):
        zerof[i] = int(bool(np.abs(rout[i]) < 1))
    sf = 1.0 - (zerof*rout)**2
    warp = warp/sf + warp*(zerof - 1)
    return warp

def nodes_2d(N):
    #compute (x,y) nodes in equilateral triangle for polynomia of order N
    alpopt = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]
    
    if N < 16:
        alpha = alpopt[N-1]
    else:
        alpha = 5/3
    
    Np = int((N+1)*(N+2)/2)

    L1 = np.zeros(Np)
    L2 = np.zeros(Np)
    L3 = np.zeros(Np)
    sk = 0
    for n in range(1,N+2):
        for m in range(1,N+3-n):
            L1[sk] = (n-1)/N
            L3[sk] = (m-1)/N
            sk = sk+1
    print(L1)
    print(L2)
    print(L3)
    L2 = 1.0 - L1 - L3
    x = -L2+L3
    y = (-L2-L3+2*L1)/np.sqrt(3.0)
    blend1 = 4*L2*L3
    blend2 = 4*L1*L3
    blend3 = 4*L1*L2

    warpf1 = warpfactor(N, L3-L2)
    warpf2 = warpfactor(N, L1-L3)
    warpf3 = warpfactor(N, L2-L1)

    warp1 = blend1*warpf1*(1+(alpha*L1)**2)
    warp2 = blend2*warpf2*(1+(alpha*L2)**2)
    warp3 = blend3*warpf3*(1+(alpha*L3)**2)

    x = x + 1*warp1 + np.cos(2*np.pi/3)*warp2 + np.cos(4*np.pi/3)*warp3
    y = y + 0*warp1 + np.sin(2*np.pi/3)*warp2 + np.sin(4*np.pi/3)*warp3

    return x,y

def rs_to_ab(r,s):
    # transfer from (r,s) -> (a,b) coordinates in triangle
    Np = len(r)
    a = np.zeros(Np)

    for n in range(len(Np)):
        if s[n] != 1:
            a[n] = 2*(1+r[n])/(1-s[n])-1
        else: 
            a[n] = -1
    b = s 
    return a,b


def xy_to_rs(x,y):
    # from (x,y) in equilateral triangle to (r,s) coordinate in standard triangle

    L1 = (np.sqrt(3.0)*y+1.0)/3.0
    L2 = (-3.0*x - np.sqrt(3.0)*y + 2.0)/6.0
    L3 = ( 3.0*x - np.sqrt(3.0)*y + 2.0)/6.0
    r = -L2 + L3 - L1
    s = -L2 - L3 + L1

    return r,s

def simplex_2d_p(a,b,i,j):
    #evaluate 2D orthonormal polynomial on simplex at(a,b) of order (i,j)
    h1 = jacobi_p(a,0,0,i)
    h2 = jacobi_p(b,2*i +1,0, j)
    P = np.sqrt(2.0)*h1*h2*(1-b)**2
    return P

def vandermonde_2d(N,r,s):
    # initialize the 2D Vandermonde matrix V_ij = phi_j(r_i,s_i)
    V2D = np.zeros((len(r),(N+1)*(N+2)/2))
    # Transfer to (a,b) coordinates
    a, b = rs_to_ab(r, s)
    # build the Vandermonde matrix
    sk = 0
    for i in range(N+1):
        for j in range(0,N-i+1):
            column = simplex_2d_p(a,b,i,j)
            for k in range(len(column)):
                V2D[k,sk]= column[k]
            sk = sk+1
 
    return V2D
def grad_jacobi_p(r,alpha,beta,N):
    #evaluate the derivative of the Jacobi polynomial of type (alpha, bet)>- 1 at points r for order N
    if N == 0:
        return np.zeros(len(r))
    else: 
        return np.sqrt(N*(N+alpha+beta+1))*jacobi_p(r, alpha+1, beta+1, N-1)

def grad_simplex_2d_p(a,b,id,jd):
    #return the derivatives of the modal basis (id,jd) on the 2D simplex at (a,b)
    fa = jacobi_p(a,0,0,id)
    dfa = grad_jacobi_p(a,0,0,id)
    gb = jacobi_p(b,2*id+1,0,jd)
    dgb = grad_jacobi_p(b,2*id+1,0,jd)

    dmodedr = dfa*gb
    if id>0:
        dmodedr = dmodedr*((0.5*(1-b))**(id-1))
    
    dmodeds = dfa*(gb*(0.5*(1+a)))
    if id>0:
        dmodeds = dmodeds*((0.5*(1-b))**(id-1))

    tmp = dgb*((0.5*(1-b))**id)
    if id>0:
        tmp = tmp-0.5*id*gb*((0.5*(1-b))**(id-1))
    dmodeds = demodes + fa*tmp
    dmodedr = 2**(id+0.5)*dmodedr
    dmodeds = 2**(id+0.5)*dmodeds

    return dmodedr, dmodeds

def grad_vandermonde_2d(N, r,s):
    # initialize the gradient of the modal basis (i,j) at (r,s) at order N
    
    a,b =  rs_to_ab(r,s)
    Np = int((N+1)*(N+2)/2)
    V2Dr = np.zeros((len(r),Np))
    V2Ds = np.zeros((len(r),Np))
    sk = 0
    for i in range(N+1):
        for j in range(N-i+1):
            column_V2Dr, column_V2Ds = grad_simplex_2d_p(a,b,i,j)
            for k in range(len(column_V2Dr)):
                V2Dr[k,sk] = column_V2Dr[k]
                V2Ds[k,sk] = column_V2Ds[k]
            sk += 1
    
    return V2Dr, V2Ds

def d_matrices_2d(N,r,s,V):
    # initialize the (r,s) differentiation matrices on the simplex, evaluated at (r,s at order N)
    Vr, Vs = grad_vandermonde_2d(N,r,s)
    Vinv = np.linalg.inv(V)
    Dr = np.matmul(Vr, Vinv)
    Ds = np.matmul(Vs, Vinv)
    return Dr, Ds

def filter_2d(Norder, Nc,sp):
    #initialize 2D filter matrix of order sp and cutoff Nc
    return
def for_global_testing():
    print (p)        










