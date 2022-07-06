
""" 
The distributions of particles require random library in numpy to work
"""
import numpy as np

class Distributions:

    def __init__(self, number_of_particles, pos):
        self.n = number_of_particles
        self.x = pos
    
    @property
    def getN(self):
        return self.n

    @property
    def getPos(self):
        return self.x

    @getN.setter
    def setN(self, n):
        self.n = n

    @getPos.setter
    def setPos(self, pos):
        self.x = pos
        
    def __add__(self, other):
        return Distributions(self.getN() + other.getN(), np.concatenate(self.getPos(), other.getPos()))

class Uniform_Spherical_Dist(Distributions):

    def __init__(self, number_of_particles, position_of_the_center, radius):
        self.n = number_of_particles
        self.p = position_of_the_center
        self.r = radius

    @property
    def getPos(self):
        
        rad = self.r**2 * np.random.random((self.n,1))
        theta = 2 * np.pi * np.random.random((self.n,1))
        phi  = np.pi * np.random.random((self.n,1))

        x =  np.sqrt(rad)*np.cos(theta)*np.sin(phi) + self.p[0]
        y =  np.sqrt(rad)*np.sin(theta)*np.sin(phi) + self.p[1]
        z =  np.sqrt(rad)*np.cos(phi) + self.p[2]
        
        r = np.hstack(x,y,z)

        return r

class Velocity_Uniform_Dist(Distributions):

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

    @property
    def getVel(self):

        vx = np.random.random(self.n,1)
        vy = np.random.random(self.n,1)
        vz = np.random.random(self.n,1)

        v = np.hstack(vx,vy,vz)

        return v


    


