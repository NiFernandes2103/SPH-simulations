from cmath import pi
import numpy as np
import matplotlib.pyplot as plt


"""
First relativistic implementation of sph method with ideal relativistic fluids


"""

"""
P = (gamma - 1)nu 

for ideal gasses

     / e  0  0  0 \
    |  0  P  0  0  |
T = |  0  0  P  0  |
     \ 0  0  0  P /
               
The initial conditions are the mass, position, velocity, 
and thermal energy of each particle
"""

#metric
nu = np.array([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) 

def __W( x, y, z, h):

    # Gaussian smoothing kernel
    r = np.sqrt(x**2 + y**2 + z**2)
    w = np.sqrt(1.0/pi)/h * np.exp(-(r/h)**2)
    
    return w


#gradient of the kernel
def gradW(x, y, z, h):
    
    r = np.sqrt(x**2 + y**2 + z**2) 

    n = -2 * np.exp(-(r/h)**2) / h**5 / (pi)**(3/2)
    wx = n*x
    wy = n*y
    wz = n*z

    return wx, wy, wz

def getPressure(x,k,T,h,mu=0.1,mh=0.001):

    """
    Equation of state for pressure of an ideal gas
    """

    rho = getDensity(x,x,mu,h)

    P = rho* k * T / (mu * mh)

    return P

def getPairwiseSeparation(xi, xj):

    """
    Get pairwise desprations between 2 sets of coordinates
    ri    is an M x 4 matrix of positions
    rj    is an N x 4 matrix of positions
    dt, dx, dy, dz   are M x N matrices of separations
    """

    M = xi.shape[0]
    N = xj.shape[0]

    # position xi = (t,x,y,z)
    
    xi1 = xi[:,0].reshape((M,1))
    xi2 = xi[:,1].reshape((M,1))
    xi3 = xi[:,2].reshape((M,1))

   
    xj1 = xj[:,0].reshape((N,1))
    xj2 = xj[:,1].reshape((N,1))
    xj3 = xj[:,2].reshape((M,1))
    
    dx = xi1 - xj1.T
    dy = xi2 - xj2.T
    dz = xi3 - xj3.T
   

    return dx,dy,dz


def W(vx, vy, vz):
    
    #Gives the gamma factor in SR for the particle with
    # v = (vx,vy,vz)

    gamma = 1/np.sqrt(1-(vx**2 + vy**2 + vz**2))

    return gamma

def getL( ri, rj, g = nu):

    L = np.sqrt(g[:,1:,1:,1:] * ri[1:].T * rj[1:])

    return L

def getDensity(r, pos, m, h):
    """
    Get Density at sampling loctions from SPH particle distribution
    r     is an M x 3 matrix of sampling locations
    pos   is an N x 3 matrix of SPH particle positions
    m     is the particle mass
    h     is the smoothing length
    rho   is M x 1 vector of density
     """

    M = r.shape[0]

    dx, dy, dz = getPairwiseSeparation(r, pos)

    rho = np.sum(m*(__W(dx,dy,dz,h)), 1).reshape((M,1))

    return rho


def getNumberDensity(r, pos, v, h):
    """
    Get Density at sampling loctions from SPH particle distribution
    r     is an M x 3 matrix of sampling locations
    pos   is an N x 3 matrix of SPH particle positions
    v     is the particle velocity
    h     is the smoothing length
    n     is the M x 1 vector of number density
     """

    M = r.shape[0]

    dx, dy, dz = getPairwiseSeparation(r, pos)

    n = np.sum(v*(__W(dx,dy,dz,h)), 1).reshape((M,1))

    return n
  

def getS(x, v, T, k, m, h):
    """
    Calculate the momentum per baryon on each SPH particle
    x     is an N x 3 matrix of 4-positions  1 to 3 entries
    u     is an N x 3 matrix of 4-velocities 1 to 3 entries
    m     is the particle mass
    h     is the smoothing length
    S     is N x 3 matrix of momentum per baryon
    """

    A = x.shape[0]
    B = v.shape[0]

    n = getNumberDensity( x, x, m, h)

    v1 = v[:,0].reshape((B,1))
    v2 = v[:,1].reshape((B,1))
    v3 = v[:,2].reshape((B,1))
    N = np.sum(n*W(v1,v2,v3))

    P = getPressure(n*m,k,T,h)

    dx, dy, dz = getPairwiseSeparation(x, x)
    wx, wy, wz = gradW( dx, dy, dz, h)

    sx = - np.sum(v.T*(P/ N**2 + P.T / N.T**2)*wx, 1).reshape((A,1))
    sy = - np.sum(v.T*(P/ N**2 + P.T / N.T**2)*wy, 1).reshape((A,1))
    sz = - np.sum(v.T*(P/ N**2 + P.T / N.T**2)*wz, 1).reshape((A,1))
    
    s = np.hstack((sx,sy,sz))

    """ Dissipative corrections should be made here"""

    # s -= lmbda * x
    
    # s -= nu*vel

    return s

def getChangeE(x, v, T, rho, k, m, h):
    """
    Calculate the momentum per baryon on each SPH particle
    x     is an N x 3 matrix of 4-positions  1 to 3 entries
    u     is an N x 3 matrix of 4-velocities 1 to 3 entries
    m     is the particle mass
    h     is the smoothing length
    de     is N x 3 matrix of momentum per baryon
    """

    A = x.shape[0]
    B = v.shape[0]


    n = getNumberDensity( x, x, m, h)

    v1 = v[:,0].reshape((B,1))
    v2 = v[:,1].reshape((B,1))
    v3 = v[:,2].reshape((B,1))
    N = np.sum(n*W(v1,v2,v3))

    P = getPressure(rho,k,T,h)

    dx, dy, dz = getPairwiseSeparation(x, x)
    wx, wy, wz = gradW( dx, dy, dz, h)

    dex = - np.sum(m*(P*v / N**2 + P.T * v.T / N.T**2)*wx, 1).reshape((A,1))
    dey = - np.sum(m*(P*v / N**2 + P.T * v.T / N.T**2)*wy, 1).reshape((A,1))
    dez = - np.sum(m*(P*v / N**2 + P.T * v.T / N.T**2)*wz, 1).reshape((A,1))
    
    de = np.hstack((dex,dey,dez))

    """ Dissipative corrections should be made here"""

    # de -= lmbda * x
    
    # de -= nu*v

    return de

def vector_Current_J(r,v,h):
    '''
    r is the position vector 
    v is the velocity vector 
    h is the smoothing parameter

    This function computes the vector current 

    '''
    dx, dy, dz = getPairwiseSeparation(r, r)
    J = np.sum(v*(v*dtaudt(v, v.T))*__W(dx,dy,dz,h))

def dtaudt(vi, vj, g_mu_nu = nu):
    '''
    vi is the velocity vector
    vj is the transpose of vi
    g_mu_nu is the metric - it is already configured as the standard metric

    The d(tau)/dt derivative is solved by this function
    '''

    dtaudt = np.sqrt(-g_mu_nu*vi*vj)
    return dtaudt

