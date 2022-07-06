
from ideal_relativistic_fluid_simulation_using_sph_method import *
import Distributions

''' here I will create tests to see if my code is working 

    I will test conservation laws
    Causality and other important 
    parameters of the simulation

'''

def test_Ideal_Hydro():

    #Simulation parameters
    N = 1000 # number of particles
    t = 0 # time
    tEnd = 15 # time which the simulation ends
    dt = 0.05 # timestep
    M = 20 # star mass
    R = 0.75 # ball radius
    h = 0.1 # smoothing length
    k = 0.1 # equation of state constant
    mu = 1 # mass of the baryon
    mh = 0.001 # mass of hidrogen

    plotRealTime = True 

    # Generate initial conditions
    np.random.seed(21)

    # lmbda is the external force the particles are subjected to
    # it is not being used for now
    # lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2 + n)/R**3/gamma(1+n))**(1/n)

    # mass per baryon
    m = M/N

    # Define the initial distributions of the variables
    pos = Distributions.Uniform_Spherical_Dist(N, np.array([0,0,0]), 1/2).getPos()
    vel = Distributions.Velocity_Uniform_Dist(N,pos).getVel()
    T = np.ones(pos.shape)

    #get the momentum per baryon
    s = getS(pos, vel, T, k, m, h)

    # number of time steps
    Nt = int(np.ceil(tEnd/dt))

    # prep the figure
    fig = plt.figure(figsize=(4,5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2,0])
    ax2 = plt.subplot(grid[2,0])
    rr = np.zeros((100,3))
    rlin = np.linspace(0,1,100)
    rr[:,0]=rlin
    #rho_analytic =lmbda/(4*k) * (R**2 -rlin**2)

    # loop for generating the simulation animation 
    # may not be used 
    for i in range(Nt):

        ''' update the parameters'''

        vel += s*dt/(2*m)

        pos += vel*dt

        s = getS(pos, vel, T, k, m, h)

        vel += s*dt/2
        
        t += dt

        n = getNumberDensity( pos, pos, m , h)

        '''May use a function to replace this code
        # Maybe an update funtion 
        '''


        '''Plot and animation'''
        if plotRealTime or (i ==Nt-1):
            plt.sca(ax1)
            plt.cla()
            cval = np.minimum((n-3)/3,1).flatten()
            plt.scatter(pos[:,0], pos[:,1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5)
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
            plt.plot(rlin, n, color='gray', linewidth=2)
            rho_radial = getNumberDensity(rr, pos, m, h)
            plt.plot(rlin, rho_radial, color='blue')
            plt.pause(0.001)
    
    plt.sca(ax2)
    plt.xlabel('radius')
    plt.ylabel('density')

    plt.savefig('sph.png',dpi=240)
    plt.show()

    return 0



