

import os
from math import pi

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

import Distributions


'''example program of sph simulation'''

# kernels
class kernel:
    
    def W( x, y, z, h):

        # Gaussian smoothing kernel
        r = np.sqrt(x**2 + y**2 + z**2)
        w = np.sqrt(1.0/pi)/h * np.exp(-(r/h)**2)
    
        return w

    def _W(x, y, z,h, v = 3):
    
        # different Kernel

        r = np.sqrt(x**2 + y**2 + z**2)
        omega = 1/ pi

        dx, dy, dz = getPairwiseSeparation(r, r)

        q = np.hstack((dx,dy,dz))
        
        if v == 3:
            if 0 < r/h and r/h < 1:
                return (omega/(h**v))*(1- 3/2 * q**2 + 3/4 * q**3)
            elif 1 < r/h and r/h < 2:
                return (omega/(h**v))*(1/4 *(2-q)**3)
            else:
                return 0
        
        elif v == 2:
            omega = 10/(7*pi)
            if 0 < r/h and r/h < 1:
                return (omega/(h**v))*(1- 3/2 * q**2 + 3/4 * q**3)
            elif 1 < r/h and r/h < 2:
                return (omega/(h**v))*(1/4 *(2-q)**3)
            else:
                return 0
        
        elif v == 1:
            omega = 2/3
            if 0 < r/h and r/h < 1:
                return (omega/(h**v))*(1- 3/2 * q**2 + 3/4 * q**3)
            elif 1 < r/h and r/h < 2:
                return (omega/(h**v))*(1/4 *(2-q)**3)
            else:
                return 0 

    #gradient of the kernel
    def gradW(x, y, z, h):
        
        r = np.sqrt(x**2 + y**2 + z**2) 

        n = -2 * np.exp(-(r/h)**2) / h**5 / (pi)**(3/2)
        wx = n*x
        wy = n*y
        wz = n*z

        return wx, wy, wz

# function that gets the distance between two particles
def getPairwiseSeparation(ri, rj):
    """
    Get pairwise desprations between 2 sets of coordinates
    ri    is an M x 3 matrix of positions
    rj    is an N x 3 matrix of positions
    dx, dy, dz   are M x N matrices of separations
    """

    M = ri.shape[0]
    N = rj.shape[0]

    # position ri = (x,y,z)
    ri1 = ri[:,0].reshape((M,1))
    ri2 = ri[:,1].reshape((M,1))
    ri3 = ri[:,2].reshape((M,1))

    rj1 = rj[:,0].reshape((N,1))
    rj2 = rj[:,1].reshape((N,1))
    rj3 = rj[:,2].reshape((N,1))
    
    dx = ri1 - rj1.T
    dy = ri2 - rj2.T
    dz = ri3 - rj3.T

    return dx,dy,dz

# gets density of the fluid at specified position
def getDensity(r, pos, m, h):
    """
    Get Density at sampling loctions from SPH particle distribution
    r     is an M x 3 matrix of sampling locations
    pos   is an N x 3 matrix of SPH particle positions
    m     is the particle mass
    h     is the smoothing length
    rho   is M x 1 vector of accelerations
     """

    M = r.shape[0]

    dx, dy, dz = getPairwiseSeparation(r, pos)

    rho = np.sum(m*(kernel.W(dx,dy,dz,h)), 1).reshape((M,1))

    return rho

# gets pressure at any time
def getPressure(rho, k , n):
    """
    Equation of State
    rho   vector of densities
    k     equat  ion of state constant
    n     polytropic index
    P     pressure
    """
    P = k * rho ** (1 + 1/n)

    return P

# function that gets the dissipation corretions
def PI(pos, vel, rho, alpha, beta,h):

    nu = np.sqrt(0.01*h**2)

    rho_ab = (rho + rho.T)/2
    v_ab = vel - vel.T
    r_ab = pos - pos.T
    r_v = np.dot(v_ab, r_ab)
    __nu__ = h*r_v/(np.dot(r_ab,r_ab)+ nu**2)

    if(r_v < 0):
        PI = -(alpha*__nu__ + beta*__nu__**2)/ rho_ab
    else:
        PI = 0
    return PI

# solves for dv/dt
def getAcc( pos, vel, m, h, k, n, lmbda, nu,viscosity=0):
    """
    Calculate the acceleration on each SPH particle
    pos   is an N x 3 matrix of positions
    vel   is an N x 3 matrix of velocities
    m     is the particle mass
    h     is the smoothing length
    k     equation of state constant
    n     polytropic index
    lmbda external force constant
    nu    viscosity
    a     is N x 3 matrix of accelerations
    """

    N = pos.shape[0]

    rho = getDensity( pos, pos, m , h)

    P = getPressure(rho, k, n)

    dx, dy, dz = getPairwiseSeparation(pos, pos)
    wx, wy, wz = kernel.gradW( dx, dy, dz, h)

    if viscosity == 1:
        ax = - np.sum(m*(P/ rho**2 + P.T / rho.T**2 + PI(pos,vel,rho,1,2,h))*wx, 1).reshape((N,1))
        ay = - np.sum(m*(P/ rho**2 + P.T / rho.T**2 + PI(pos,vel,rho,1,2,h))*wy, 1).reshape((N,1))
        az = - np.sum(m*(P/ rho**2 + P.T / rho.T**2 + PI(pos,vel,rho,1,2,h))*wz, 1).reshape((N,1))
    else:
        ax = - np.sum(m*(P/ rho**2 + P.T / rho.T**2)*wx, 1).reshape((N,1))
        ay = - np.sum(m*(P/ rho**2 + P.T / rho.T**2)*wy, 1).reshape((N,1))
        az = - np.sum(m*(P/ rho**2 + P.T / rho.T**2)*wz, 1).reshape((N,1))

    a = np.hstack((ax,ay,az))

    a -= lmbda * pos
    
    # a -= nu*vel

    return a

# alternative way to get density
def continuityEquation(r, pos, va, vb, m, h):

    M = r.shape[0]

    dx, dy, dz = getPairwiseSeparation(r, pos)

    drho = np.sum(m * np.abs(va - vb) * kernel.gradW(dx,dy,dz,h)).reshape((M,1))

    return drho

# solves for velocity divergence
def getVelocityDivergence(r, pos, va, vb, h):
    pass
    #dx, dy, dz = getPairwiseSeparation(r, pos)
    #deltaV =  np.sum




def main():

    #Simulation parameters
    N = 100 # number of particles
    t = 0 # time
    tEnd = 12 # time which the simulation ends
    dt = 0.05 # timestep
    M = 2 # star mass
    R = 0.75 # star radius
    h = 0.1 # smoothing length
    k = 0.1 # equation of state constant
    n = 1 # polytropic index
    nu = 1 # damping
    plotRealTime = True 

    # Generate initial conditions
    np.random.seed(21)

    lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2 + n)/R**3/gamma(1+n))**(1/n)
    m = M/N

    star1 = Distributions.Uniform_Spherical_Dist(N/2, np.array([-1/2,0,0]), 1/4)
    star2 = Distributions.Uniform_Spherical_Dist(N/2, np.array([1/2,0,0]), 1/4)
    pos = star1 + star2
    vel = np.zeros(pos.shape) 

    acc = getAcc(pos, vel, m, h, k, n, lmbda, nu)

    Nt = int(np.ceil(tEnd/dt))

    # prep the figure
    fig = plt.figure(figsize=(4,5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2,0])
    ax2 = plt.subplot(grid[2,0])
    rr = np.zeros((100,3))
    rlin = np.linspace(0,1,100)
    rr[:,0]=rlin
    rho_analytic =lmbda/(4*k) * (R**2 -rlin**2)

    for i in range(Nt):

        vel += acc*dt/2

        pos += vel*dt

        acc = getAcc(pos, vel, m, h, k, n, lmbda, nu)

        vel += acc*dt/2
        
        t += dt

        rho = getDensity( pos, pos, m , h)

        if plotRealTime or (i ==Nt-1):
            plt.sca(ax1)
            plt.cla()
            cval = np.minimum((rho-3)/3,1).flatten()
            plt.scatter(pos[:,0], pos[:,1], c=cval , cmap=plt.cm.autumn, alpha=0.5)
            ax1.set(xlim=(-1.4,1.4), ylim=(-1.2,1.2))
            ax1.set_aspect('equal','box')
            ax1.set_xticks([-1,0,1])
            ax1.set_yticks([-1,0,1])
            ax1.set_facecolor('black')
            ax1.set_facecolor((.1,.1,.1))

            plt.sca(ax2)
            plt.cla()
            ax2.set(xlim=(0,1), ylim=(0,3))
            ax2.set_aspect(0.1)
            plt.plot(rlin, rho_analytic, color='gray', linewidth=2)
            rho_radial = getDensity(rr, pos, m, h)
            plt.plot(rlin, rho_radial, color='blue')
            plt.pause(0.001)
    
    plt.sca(ax2)
    plt.xlabel('radius')
    plt.ylabel('density')

    #plt.savefig('sph.png',dpi=240)
    
    plt.show()

    return 0

if __name__ == "__main__":
    main()


