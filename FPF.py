"""
Created on Wed Jan. 23, 2019

@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from scipy.signal import find_peaks

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
    def __init__(self, number_of_particles, f_min, f_max, dt):
        """Initialize Particles"""
        self.dt = dt
        self.N = int(number_of_particles)
        self.theta = np.reshape(np.linspace(0, 2*np.pi, self.N, endpoint=False), [-1,1])
        self.omega_bar = 2*np.pi*np.reshape(np.linspace(f_min, f_max, self.N), [-1,1])
        self.omega = 2*np.pi*np.reshape(np.linspace(f_min, f_max, self.N), [-1,1])
        self.sync_cutoff = np.exp(-((f_max-f_min)/(self.N-1))**2)
        self.amp = np.reshape(np.ones(self.N), [-1,1])
        self.h = np.reshape(np.zeros(self.N), [-1,1])
        self.theta_error_sum = np.reshape(np.zeros(self.N), [-1,1])
        self.Kp = 1
        self.Ki = 0.01
        self.sync_matrix = np.zeros([self.N, self.N])
        self.update_sync_matrix()

    
    def update(self, theta_error):
        """mod the angle of each particle with 2pi"""
        # self.theta = np.mod(self.theta, 2*np.pi)
        """update omega"""
        self.theta_error_sum = theta_error*self.dt + 0.9*self.theta_error_sum
        self.omega += self.Kp*theta_error+self.Ki*self.theta_error_sum
        return
        
    def update_sync_matrix(self):
        """update sync_matrix"""
        omega = np.sort(self.omega[:,0])
        for ni in range(self.N):
            for nj in range(ni, self.N):
                self.sync_matrix[ni,nj] = np.exp(-(omega[ni]-omega[nj])**2)
                self.sync_matrix[nj,ni] = self.sync_matrix[ni,nj]
        return self.sync_matrix

    def check_sync(self):
        """check synchronized frequencies"""
        self.update_sync_matrix()
        sync_particles = np.zeros(2*self.N-1)
        for i in range(sync_particles.shape[0]):
            if i < self.N:
                sync_vector = np.zeros(int(i/2)+1)
            else:
                temp_i = 2*self.N-2-i
                sync_vector = np.zeros(int(temp_i/2)+1)
            # Calculate sync_vector
            for j in range(sync_vector.shape[0]):
                sync_vector[j] = self.sync_matrix[int(i/2)+j+np.mod(i,2),int(i/2)-j]    
            # Find the farest sync_particles
            for j in range(sync_vector.shape[0]):
                if sync_vector[j] > self.sync_cutoff:
                    sync_particles[i] =  j
        # Add bound flags at the Head and the Tail
        sync_particles = np.append(-np.insert(sync_particles,0,1),[-1])
        # Filter wripples
        for i in range(sync_particles.shape[0]-2):
            if sync_particles[i] == sync_particles[i+2]:
                sync_particles[i+1] = sync_particles[i]
        
        ## Find Peaks and Cutoff Frequencies
        indices, _ = find_peaks(sync_particles)
        cutoff_sync_freq = np.array([self.omega[int((index+1)/2),0] for index in indices])
        sync_index = np.zeros([cutoff_sync_freq.shape[0]-1,2])
        sync_freq = np.zeros([cutoff_sync_freq.shape[0]-1,2])
        for i in range(sync_freq.shape[0]):
            sync_index[i,:] = (indices[i:i+2]+1)/2
            sync_freq[i,:] = cutoff_sync_freq[i:i+2]

        print(sync_freq/(2*np.pi), sync_index)
        return sync_particles
        # return sync_freq/(2*np.pi), sync_index

    def get_theta(self):
        return np.squeeze(self.theta)

    def get_amp(self):
        return np.squeeze(self.amp)
    
    def get_freq(self):
        return np.squeeze(self.omega/(2*np.pi))

    def get_freq_range(self):
        return np.squeeze(self.omega_bar/(2*np.pi))

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
        self.particles = Particles(number_of_particles=number_of_particles, f_min=f_min, f_max=f_max, dt=dt) 
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
        dtheta = self.particles.omega_bar*self.dt + self.ito_integral(dI=dI, dh=self.particles.h-self.h_hat)
        self.particles.theta += dtheta
        self.particles.amp -= -1*(y-self.particles.h)*2*self.h(1,self.particles.theta)*self.dt          #update amp indivisually
        # self.particles.amp -= -1*(y-self.h_hat)*2*np.mean(self.h(1,self.particles.theta))*self.dt       #update amp together
        self.h_hat = self.calculate_h_hat()
        self.particles.update(theta_error=dtheta-self.particles.omega*self.dt)
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
            h(amp, x): observation function maps states(X) to observations(Y)
    """
    def __init__(self, freq, amp, sigma_B, sigma_W, dt, T):
        """Initialize"""
        self.omega = 2*np.pi*(np.array(freq))
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

    def h(self, amp, x):
        """observation function maps states(x) to observations(y)"""
        return np.matmul(amp, np.sin(x))

def h(amp, x):
        return amp*np.cos(x)

if __name__ == "__main__":
    # N states of frequency inputs
    freq = [1.]
    # M-by-N amplitude matrix
    amp = [[10]]
    # N states of state noises
    sigma_B = [0.1]
    # M states of signal noises
    sigma_W = [0.1]

    T = 10.
    dt = 0.001
    signal = Signal(freq=freq, amp=amp, sigma_B=sigma_B, sigma_W=sigma_W, dt=dt, T=T)
    h_hat = np.zeros(signal.Y.shape)
    
    N=100
    feedback_particle_filter = FPF(number_of_particles=N, f_min=0.9, f_max=1.1, sigma_W=sigma_W, dt=dt, h=h)
    theta = np.zeros([N, signal.Y.shape[1]])
    amp = np.zeros([N, signal.Y.shape[1]])
    freq_hat = np.zeros([N, signal.Y.shape[1]])

    for k in range(signal.Y.shape[1]):
        feedback_particle_filter.update(signal.Y[:,k])
        h_hat[:,k] = feedback_particle_filter.h_hat
        theta[:,k] = feedback_particle_filter.particles.get_theta()
        amp[:,k] = np.squeeze(feedback_particle_filter.particles.amp)
        freq_hat[:,k] = feedback_particle_filter.particles.get_freq()

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fontsize = 20
    fig, axes = plt.subplots(signal.Y.shape[0],1, figsize=(9,7))
    try:
        iterator = iter(axes)
    except TypeError:
        ax = axes
        ax.plot(signal.t, signal.Y[0], label=r'signal')
        ax.plot(signal.t, h_hat[0,:], label=r'estimation')
    else:
        for i, ax in enumerate(axes):
            ax.plot(signal.t, signal.Y[i], label=r'signal')
            ax.plot(signal.t, h_hat[i,:], label=r'estimation')
    ax.legend(fontsize=fontsize-5)
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('time', fontsize=fontsize)

    fig, ax = plt.subplots(1,1, figsize=(9,7))
    for k in range(N):
        ax.scatter(signal.t, theta[k,:], s=0.5)
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('time', fontsize=fontsize)
    ax.set_ylabel(r'$\theta$ [rad]', fontsize=fontsize)
    ax.set_title('Particles', fontsize=fontsize+2)
    
    fig, ax = plt.subplots(1,1, figsize=(9,7))
    for k in range(N):
        ax.plot(signal.t, amp[k,:]) 
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('time', fontsize=fontsize)
    ax.set_ylabel('$A$', fontsize=fontsize)
    ax.set_title('Amplitude of Particles', fontsize=fontsize+2)
        
    fig, ax = plt.subplots(1,1, figsize=(9,7))
    for k in range(N):
        ax.plot(signal.t, freq_hat[k,:]) 
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('time', fontsize=fontsize)
    ax.set_ylabel('$\omega$ [Hz]', fontsize=fontsize)
    ax.set_title('Frequency of Particles', fontsize=fontsize+2)
    
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    ax.pcolor(range(N), range(N), feedback_particle_filter.particles.update_sync_matrix(), cmap='gray', vmin=0, vmax=1)
 
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    ax.plot(feedback_particle_filter.particles.check_sync())
    
    
    plt.show()

