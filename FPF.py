"""
Created on Wed Jan. 23, 2019

@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
import copy

class Particles(object):
    """
    Particles: particles has a scalar state which denotes phase in [0 2pi] cycle.
        Initialize = Particles(number_of_particles, f_min, f_max)
            number_of_particles: integer, number of particles
            f_min: float, minimum frequency in Hz
            f_max: float, maximum frequency in Hz
        Members =
            N: integer, number of particles
            theta: numpy array with the shape of (N,1), the angle (rad) for N particles
            omega: numpy array with the shape of (N,1), the inherited frequency (rad/s) for N particles
            amp: numpy array with the shape of (N,1), the amplitude for N particles
            h: numpy array with the shape of (N,1), the esitmated observation for N particles
        Methods =
            update(): mod the angle of each particle with 2pi
        
    """
    def __init__(self, number_of_particles, f_min, f_max):
        """Initialize Particles"""
        self.N = int(number_of_particles)
        self.theta = np.reshape(np.linspace(0, 2*np.pi, self.N, endpoint=False), [-1,1])
        self.omega = 2*np.pi*np.reshape(np.linspace(f_min, f_max, self.N), [-1,1])
        self.amp = np.reshape(np.ones(self.N), [-1,1])
        self.h = np.reshape(np.zeros(self.N), [-1,1])
    
    def update(self):
        """mod the angle of each particle with 2pi"""
        self.theta = np.mod(self.theta, 2*np.pi)

class FPF(object):
    """
    FPF: Feedback Partilce Filter
        Notation Note:
            N: number of particles
            M: number of observations(y)
        Initialize = FPF(number_of_particles, f_min, f_max, sigma_W, dt, h)
            number_of_particles: integer, number of particles
            f_min: float, minimum frequency in Hz
            f_max: float, maximum frequency in Hz
            sigma_W: 1D list with legnth M, standard deviation of noise of observations(y)
            dt: float, size of time step in sec
            h: function, estimated observation dynamics
        Members = 
            particles: Particles
            sigma_W: numpy array with the shape of (M,1), standard deviation of noise of observations(y)
            dt: float, size of time step in sec
            h: function, estimated observation dynamics
            h_hat: float, filtered observation
        Methods =
            ito_integral(dI, dh): Calculate the ito integral
            calculate_h_hat(): Calculate h for each particle and the average of h (h_hat)
            update(y): Update each particle with new observation data y
        
    """
    def __init__(self, number_of_particles, f_min, f_max, sigma_W, dt, h):
        """Initialize Particles"""
        self.particles = Particles(number_of_particles=number_of_particles, f_min=f_min, f_max=f_max) 
        self.sigma_W = np.reshape(np.array(sigma_W), [len(sigma_W),-1])
        self.dt = dt
        self.h = h
        self.h_hat = self.calculate_h_hat()
    
    def ito_integral(self, dI, dh):
        """Calculate the ito integral"""
        """
            arguments = 
                dI: numpy array with the shape of (N,M), inovation process
                dh: numpy array with the shape of (N,1), the esitmated observation for N particles minus their average
            variables = 
                K: numpy array with the shape of (N,1,M), gain for each particle
                partial_K: numpy array with the shape of (N,1,M), partial_K/partial_theta for each particle
                dI: numpy array with the shape of (N,M,1), inovation process 
        """
        def calculate_K(self, dh):
            """
            Calculate K and partial K with respect to theta
            A*kappa=b

            """
            """
                variables = 
                    K: numpy array with the shape of (N,1,M), gain for each particle
            """
            def calculate_A(self):
                A = np.zeros([2,2])
                c = np.cos(self.particles.theta)
                s = np.sin(self.particles.theta)
                A[0,0] = np.mean(s*s)
                A[0,1] = np.mean(-s*c)
                A[1,0] = A[0,1]
                A[1,1] = np.mean(c*c)
                return A
            
            def calculate_b(self, dh):
                b = np.zeros(2)
                b[0] = np.mean(dh*np.cos(self.particles.theta))
                b[1] = np.mean(dh*np.sin(self.particles.theta))
                return b
            
            A = calculate_A(self)
            b = calculate_b(self, dh)
            K = np.zeros([self.particles.theta.shape[0], self.sigma_W.shape[0]])
            partial_K = np.zeros(K.shape)
            for m, sigma_W in enumerate(self.sigma_W):
                kappa = np.linalg.solve(A,b/sigma_W)
                K[:,m] = np.squeeze(-kappa[0]*np.sin(self.particles.theta) + kappa[1]*np.cos(self.particles.theta))
                partial_K[:,m] = np.squeeze(-kappa[0]*np.cos(self.particles.theta) - kappa[1]*np.sin(self.particles.theta))
            K = np.reshape(K, [K.shape[0], 1, K.shape[1]])
            partial_K = np.reshape(K, K.shape)
            return K, partial_K
        
        K, partial_K = calculate_K(self, dh)
        dI = np.reshape(dI, [dI.shape[0], dI.shape[1], 1])
        return np.reshape(np.matmul(K, dI) + np.matmul(0.5*K*partial_K*self.dt,self.sigma_W**2), [-1,1])

    def calculate_h_hat(self):
        """Calculate h for each particle and the average of h (h_hat)"""
        self.particles.h = self.h(self.particles.amp, self.particles.theta)
        h_hat = np.mean(self.particles.h, axis=0)
        return h_hat

    def update(self, y):
        """Update each particle with new observation data y"""
        """
            argument =
                y: numpy array with the shape of (M,), new observation data
            variables =
                dz: numpy array with the shape of (1,M), integral of y over time period dt
                dI: numpy array with the shape of (N,M), inovation process dI = dz-0.5*(h+h_hat)*dt for each particle
        """
        dz = np.reshape(y*self.dt, [1,y.shape[0]])
        dI = dz-np.repeat(0.5*(self.particles.h+self.h_hat)*self.dt, repeats=dz.shape[1], axis=1)
        self.particles.theta += self.particles.omega*self.dt + self.ito_integral(dI=dI, dh=self.particles.h-self.h_hat)
        self.particles.update()
        self.h_hat = self.calculate_h_hat()
        return

class Signal(object):
    """
    Signal:
        Notation Notes:
            N: number of states(X)
            M: number of observations(Y)
        Initialize = Signal(freq, amp, sigma_B, sigma_W, dt, T)
            freq: 1D list with length N, list of freqencies in Hz
            amp: 2D list with shape of M-by-N, amplitude parameters for h function in Y=h(X)
            sigma_B: 1D list with legnth N, standard deviation of noise of states(X)
            sigma_W: 1D list with legnth M, standard deviation of noise of observations(Y)
            dt: float, size of time step in sec
            T: float or int, end time in sec
        Members =
            omega: numpy array with shape of (N,), list of freqencies in rad/s
            amp: numpy array with the shape of (M,N), amplitude parameters for h function in Y=h(X)
            sigma_B: numpy array with the shape of (N,1), standard deviation of noise of states(X)
            sigma_W: numpy array with the shape of (M,1), standard deviation of noise of observations(Y)
            dt: float, size of time step in sec
            T: float, end time in sec, which is also an integer mutiple of dt
            t: numpy array with the shape of (T/dt+1,), i.e. 0, dt, 2dt, ... , T-dt, T
            X: numpy array with the shape of (N,T/dt+1), states in time series
            dX: numpy array with the shape of (N,T/dt+1), states increment in time series
            Y: numpy array with the shape of (M,T/dt+1), observations in time series
        Methods =
            hz2rad(freq): chage of unit from Hz to rad/s
            h(amp, x): observation function maps states(X) to observations(Y)
    """
    def __init__(self, freq, amp, sigma_B, sigma_W, dt, T):
        """Initialize Particles"""
        self.omega = self.hz2rad(np.array(freq))
        self.amp = np.array(amp)
        self.sigma_B = np.reshape(np.array(sigma_B), [self.amp.shape[1],-1])
        self.sigma_W = np.reshape(np.array(sigma_W), [self.amp.shape[0],-1])
        self.dt = dt
        self.T = self.dt*int(T/self.dt)
        self.t = np.arange(0,self.T+self.dt,self.dt)
        self.X = np.zeros([self.sigma_B.shape[0], self.t.shape[0]])
        self.dX = np.zeros(self.X.shape)
        for n in range(self.X.shape[0]):
            """dX = omega*dt + sigma_B*dB"""
            self.dX[n,:] = self.omega[n]*self.dt + self.sigma_B[n]*self.dt*np.random.normal(0,1,self.t.shape[0])
            self.X[n,:] = np.cumsum(self.dX[n,:])
        self.Y = self.h(self.amp, self.X) + self.sigma_W*np.random.normal(0,1,[self.sigma_W.shape[0],self.t.shape[0]])
    
    def hz2rad(self, freq):
        """chage of unit from Hz to rad/s"""
        return 2*np.pi*freq

    def h(self, amp, x):
        """observation function maps states(x) to observations(y)"""
        return np.matmul(amp, np.cos(x))

def h(amp, x):
        return amp*np.cos(x)

if __name__ == "__main__":
    # N states of frequency inputs
    freq = [1., 3.]
    # M-by-N amplitude matrix
    amp = [[1,1]]
    # N states of state noises
    sigma_B = [0.1,0.1]
    # M states of signal noises
    sigma_W = [0.1]

    T = 10.
    dt = 0.01
    signal = Signal(freq=freq, amp=amp, sigma_B=sigma_B, sigma_W=sigma_W, dt=dt, T=T)
    h_hat = np.zeros(signal.Y.shape)
    
    N=100
    feedback_particle_filter = FPF(number_of_particles=N, f_min=0.9, f_max=1.6, sigma_W=sigma_W, dt=dt, h=h)
    theta = np.zeros([N, signal.Y.shape[1]])

    for k in range(signal.Y.shape[1]):
        feedback_particle_filter.update(signal.Y[:,k])
        h_hat[:,k] = feedback_particle_filter.h_hat
        theta[:,k] = np.squeeze(feedback_particle_filter.particles.theta)
    
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(signal.Y.shape[0],1, figsize=(9,7))
    try:
        iterator = iter(axes)
    except TypeError:
        ax = axes
        ax.plot(signal.t, signal.Y[0])
        ax.plot(signal.t, h_hat[0,:])
    else:
        for i, ax in enumerate(axes):
            ax.plot(signal.t, signal.Y[i])
            ax.plot(signal.t, h_hat[i,:])
    
    fig, ax = plt.subplots(1,1, figsize=(9,7))
    for k in range(N):
        ax.scatter(signal.t, theta[k,:], s=0.5) 
    
    plt.show()

