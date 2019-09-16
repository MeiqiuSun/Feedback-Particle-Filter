"""
Created on Wed Jan. 23, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod

import numpy as np
import sympy

from plotly import subplots
import plotly.offline as pyo
import plotly.graph_objs as go

from Tool import Struct, isiterable, find_limits, find_closest, circle_mean, default_colors
from Signal import Signal, Sinusoids
from Particles import Particles

class Model(ABC):
    """Model:
        Notation Notes:
            Np: number of particles
        Initialize = Model(self, m, d, X0_range, states, states_label, sigma_V, sigma_W):
            m: int, number of observations
            d: int, number of states
            X0_range: 2D list of float with the length d-by-2, range of initial states
            states: list of string with the length d, name of states
            states_label:  list of string with the length d, legend name of states
            sigma_V: 1D list of float with the length d, standard deviation of process noise
            sigma_W: 1D list of float with the length m, standard deviation of observation noise
        Members:
            m: int, number of observations
            d: int, number of states
            X0_range: numpy array with the shape of (d,2), range of initial states
            states: list of string with the length d, name of states
            states_label:  list of string with the length d, legend name of states
            sigma_V: numpy array with the shape of (d,), standard deviation of process noise
            sigma_W: numpy array with the shape of (m,), standard deviation of observation noise
        Methods:
            @abs states_constraints(self, X): return an numpy array with the shape of (Np,d), modified states
            @abs f(self, X, t): return a numpy array with the shape of (Np,d), state tansition function maps states to rate change of states
            @abs h(self, X): return a numpy array with the shape of (Np,m), observation function maps states to observations
    """

    def __init__(self, m, d, X0_range, states, states_label, sigma_V, sigma_W):
        self.m = int(m)
        self.d = int(d)
        self.X0_range = np.array(X0_range)
        self.states = states
        self.states_label = states_label
        self.sigma_V = np.array(sigma_V)
        self.sigma_W = np.array(sigma_W)
    
    @abstractmethod
    def states_constraints(self, X):
        # return X
        raise NotImplementedError
    
    @abstractmethod
    def f(self, X, t):
        # dXdt = np.zeros(X.shape)
        raise NotImplementedError

    @abstractmethod
    def h(self, X):
        # Y = np.zeros(self.sigma_W.shape)
        raise NotImplementedError

class Galerkin(ABC):
    """Galerkin: An approximation in finite element method:
        Notation Notes:
            Np: number of particles
        Initialize = Galerkin(self, states, m, L):
            states: 1D list of string with the length of d, name of states
            m: int, number of observations
            L: int, number of base functions
        Members:
            states: 
                (1) in the __init__ function: 1D list of string with the length of d, name of states
                (2) after the __init__ function: 1D list of numpy array with the length of d, value of states of each particles (refer to method split(self, X))
            d: int, number of states
            m: int, number of observations
            L: int, number of base functions
            c: numpy array with the shape of (m,L), coefficients of base functions
        Methods:
            @abs psi(self, X): return a numpy array with the shape of (L,Np), base functions
            @abs grad_psi(self, X): return a numpy array with the shape of (L,d,Np), gradient of base functions
            @abs grad_grad_psi(self, X): return a numpy array with the shape of (L,d,d,Np), gradient of gradient of base functions
            split(self, X): split values of X into states
            calculate_K(self, X, normalized_dh): return two numpy arrays with the shape of (Np,d,m) and (Np,d,m,d), gain function K and gradient of K
    """

    def __init__(self, states, m, L):
        self.states = ['self.'+state for state in states]
        self.d = len(states)
        self.m = int(m)
        self.L = int(L)
        self.c = np.zeros((self.m, self.L))
        return

    @abstractmethod
    def psi(self, X):
        raise NotImplementedError("Subclasses of Galerkin must set an instance method psi(self, X)")
    
    
    @abstractmethod
    def grad_psi(self, X):
        raise NotImplementedError("Subclasses of Galerkin must set an instance method grad_psi(self, X)")

    
    @abstractmethod
    def grad_grad_psi(self, X):
        raise NotImplementedError("Subclasses of Galerkin must set an instance method grad_grad_psi(self, X)")    
    
    def split(self, X):
        for d in range(self.d):
            exec(self.states[d]+' = X[:,{}]'.format(d))
        return

    def calculate_K(self, X, normalized_dh):
        """calculate the optimal feedback gain function K by using Galerkin Approximation to solve the BVP
            Arguments:
                X: numpy array with the shape of (Np,d), the estimated states for Np particles
                normalized_dh: numpy array with the shape of (Np,m), normalized filtered observation difference between each particle and mean
            Variables:
                K: numpy array with the shape of (Np,d,m), gain for each particle
                partial_K: numpy array with the shape of (Np,d,m,d), partial_K/partial_X for each particle
                psi: numpy array with the shape of (L,Np), base functions psi
                grad_psi: numpy array with the shape of (L,d,Np), gradient of base functions psi
                grad_grad_psi: numpy array with the shape of (L,d,d,Np), gradient of gradient of base functions psi
                A: numpy array with the shape of (L,L), A_(i,j) = Exp[grad_psi_j dot grad_psi_i]
                b: numpy array with the shape of (m,L), b = Exp[normalized_dh*psi] differ from one observation to the other
                c: numpy array with the shape of (m,L), the solution to A*c=b
        """

        def calculate_A(self, grad_psi):
            """calculate the A matrix in the Galerkin Approximation in Finite Element Method
                Arguments:
                    grad_psi: numpy array with the shape of (L,d,Np), gradient of base functions psi
                Variables:
                    A: numpy array with the shape of (L,L), A_(i,j) = Exp[grad_psi_j dot grad_psi_i]
            """

            A = np.zeros([self.L, self.L])
            for li in range(self.L):
                for lj in range(li, self.L):
                    A[li,lj] = np.mean(np.sum(grad_psi[lj]*grad_psi[li], axis=0))
                    if not li==lj:
                        A[lj,li] = A[li,lj]
            # print(np.linalg.det(A))
            if np.linalg.det(A)<=0.01:
                A += np.diag(0.01*np.random.randn(self.L))
            # np.set_printoptions(precision=1)
            # print(A)
            return A
        
        def calculate_b(self, normalized_dh, psi):
            """calculate the b vector in the Galerkin Approximation in Finite Element Method
                Arguments:
                    normalized_dh: numpy array with the shape of (Np,), normalized mth channel of filtered observation difference between each particle and mean
                    psi: numpy array with the shape of (L,Np), base functions psi
                Variables:
                    b = numpy array with the shape of (L,), b = Exp[normalized_dh*psi]
            """

            b = np.zeros(self.L)
            for l in range(self.L):
                b[l] = np.mean(normalized_dh*psi[l])
            return b
        
        K = np.zeros((X.shape[0], self.d, self.m))
        partial_K = np.zeros((X.shape[0], self.d, self.m, self.d))
        grad_psi = self.grad_psi(X)            
        grad_grad_psi = self.grad_grad_psi(X)
        A = calculate_A(self, grad_psi)
        b = np.zeros((self.m, self.L))
        for m in range(self.m):
            b[m] = calculate_b(self, normalized_dh[:,m], self.psi(X))
            self.c[m] = np.linalg.solve(A,b[m])
            for l in range(self.L):
                K[:,:,m] += self.c[m,l]*np.transpose(grad_psi[l])
                partial_K[:,:,m,:] += self.c[m,l]*np.transpose(grad_grad_psi[l], (2,0,1))
        return K, partial_K

class Galerkin_sympy(Galerkin):
    """Galerkin_sympy: An approximation in finite element method:
            states: [ theta, omega, a1, a2, b2, ..., am, bm]
            base functions: [ omega, a1*cos(theta), a1*sin(theta), a2*cos(theta), b2*sin(theta), ..., am*cos(theta), bm*sin(theta)] 
        Initialize = Galerkin_sympy(states, m, psi_list)
            states: 1D list of string with length d, name of each state
            m: int, number of observations
            psi_list: 1D list of string with length L, sympy expression of each base function
        Members = 
            result: temporary memory location of the result of exec()
            psi_list: 1D list of string with length L, sympy expression of each base function
            grad_psi_list: 2D list of string with length L-by-d, sympy expression of gradient of psi function
            grad_grad_psi_list: 3D list of string with length L-by-d-by-d, sympy expression of gradient of gradient of psi function
            psi_string: string, executable 1D list of numpy array with length L
            grad_psi_string: string, executable 2D list of numpy array with length L-by-d
            grad_grad_psi_string: string, executable 3D list of numpy array with length L-by-d-by-d
        Methods = 
            sympy_executable_result(self): return sympy executable string
    """
    def __init__(self, states, m, psi_list):
        Galerkin.__init__(self, states, m, len(psi_list))

        self.psi_list = psi_list
        self.psi_string = '[' + (', '.join(self.psi_list)).replace('sympy','np') + ']'

        self.result = ''

        self.grad_psi_list = [[None for d in range(self.d)] for l in range(self.L)]
        self.grad_grad_psi_list = [[[None for d2 in range(self.d)] for d1 in range(self.d)] for l in range(self.L)]  
        
        # initialize symbols
        exec(', '.join(self.states)+' = sympy.symbols("'+', '.join(self.states)+'")')
        
        # calculate grad_psi and grad_grad_psi
        self.grad_psi_string = '['
        self.grad_grad_psi_string = '['
        for l in range(self.L):
            self.grad_psi_string += '['
            self.grad_grad_psi_string += '['
            for d1 in range(self.d):
                exec('self.result = sympy.diff('+self.psi_list[l]+','+self.states[d1]+')')
                self.grad_psi_list[l][d1] = self.sympy_executable_result()
                
                self.grad_grad_psi_string += '['
                for d2 in range(self.d):
                    exec('self.result = sympy.diff('+self.grad_psi_list[l][d1]+','+self.states[d2]+')')
                    self.grad_grad_psi_list[l][d1][d2] = self.sympy_executable_result()
                self.grad_grad_psi_string += ((', ').join(self.grad_grad_psi_list[l][d1])).replace('sympy','np')
                self.grad_grad_psi_string += '], '
            
            self.grad_psi_string += ((', ').join(self.grad_psi_list[l])).replace('sympy','np')
            self.grad_psi_string += '], '
            self.grad_grad_psi_string = self.grad_grad_psi_string[:-2] + '], '

        self.grad_psi_string = self.grad_psi_string[:-2]+']'
        self.grad_grad_psi_string = self.grad_grad_psi_string[:-2]+']'
        return

    def sympy_executable_result(self):
        self.result = str(self.result)
        if 'cos' in self.result:
            self.result = self.result.replace('cos','sympy.cos')
        if 'sin' in self.result:
            self.result = self.result.replace('sin','sympy.sin')
        return self.result

    def psi(self, X):
        self.split(X)
        exec("self.result = "+self.psi_string)
        trial_functions = self.result
        return np.array(trial_functions)
        
    def grad_psi(self, X):
        self.split(X)
        exec("self.result = "+self.grad_psi_string)
        grad_trial_functions = np.zeros([self.L, self.d, X.shape[0]])
        for l in range(self.L):
            for d in range(self.d):
                grad_trial_functions[l,d,:] = self.result[l][d]
        return grad_trial_functions

    def grad_grad_psi(self, X):
        self.split(X)
        exec("self.result = "+self.grad_grad_psi_string)
        grad_grad_trial_functions = np.zeros([self.L, self.d, self.d, X.shape[0]])
        for l in range(self.L):
            for d1 in range(self.d):
                for d2 in range(self.d):
                    grad_grad_trial_functions[l,d1,d2,:] = self.result[l][d1][d2]
        return grad_grad_trial_functions

class FPF(object):
    """FPF: Feedback Partilce Filter
        Notation Note:
            Np: number of particles
        Initialize = FPF(number_of_particles, model, galerkin, dt)
            number_of_particles: integer, number of particles
            model: class Model, f, h and other paratemers are defined at here
            galerkin: class Galerkin, base functions are defined at here
            dt: float, size of time step in sec
        Members:
            d: number of states(X)
            m: number of observations(Y)
            sigma_V: numpy array with the shape of (d,), standard deviations of process noise(X)
            sigma_W: numpy array with the shape of (m,), standard deviations of noise of observations(Y)
            f: function, state dynamics
            h: function, estimated observation dynamics
            dt: float, size of time step in sec
            particles: class Particles,
            galerkin: class Galerkin, base functions are defined at here
            get_states: function, return states (X)
            h_hat: numpy array with the shape of (m,), filtered observations
        Methods:
            optimal_control(self, dI, dZ): calculate the optimal control with new observations(Y)
            calculate_h_hat(self): Calculate h for each particle and h_hat by averaging h from all particles
            update(self, Y): Update each particle with new observations(Y)
            run(self, Y, show_time=False): Run FPF with time series observations(Y)
    """
    def __init__(self, number_of_particles, model, galerkin, dt):
        self.states_label = model.states_label
        self.d = model.d
        self.m = model.m
        self.sigma_V = model.sigma_V
        self.sigma_W = model.sigma_W
        self.f = model.f
        self.h = model.h
        self.dt = dt

        self.particles = Particles(number_of_particles=number_of_particles, \
                                    X0_range=model.X0_range, states_constraints=model.states_constraints, \
                                    m=self.m) 
        self.galerkin = galerkin
        self.get_states = model.get_states
        self.h_hat = self.calculate_h_hat()

    def optimal_control(self, dI, dZ):
        """calculate the optimal control with new observations(Y):
            Arguments = 
                dI: numpy array with the shape of (Np,m), inovation process
            Variables = 
                K: numpy array with the shape of (Np,d,m), gain for each particle
                partial_K: numpy array with the shape of (Np,d,m,d), partial_K/partial_X for each particle
        """
        K, partial_K = self.galerkin.calculate_K(self.particles.X, (self.particles.h-self.h_hat)/np.square(self.sigma_W))
        # return np.einsum('inm,m->in',K,dZ[0,:]) - np.einsum('inm,m->in',K,self.h_hat)*self.dt
        return np.einsum('idm,im->id',K,dI) + 0.5*np.einsum('ijm,idmj->id',K,np.einsum('idmj,m->idmj',partial_K,np.square(self.sigma_W)))*self.dt
    
    def calculate_h_hat(self):
        """Calculate h for each particle and h_hat by averaging h from all particles"""
        self.particles.h = np.reshape(self.h(self.particles.X), [self.particles.Np, self.m])
        h_hat = np.mean(self.particles.h, axis=0)
        return h_hat

    def update(self, Y):
        """Update each particle with new observations(Y):
            argument =
                Y: numpy array with the shape of (m,), new observation data at a fixed time
            variables =
                dZ: numpy array with the shape of ( 1,m), integral of y over time period dt
                dI: numpy array with the shape of (Np,m), inovation process dI = dz-0.5*(h+h_hat)*dt for each particle
                dX: numpy array with the shape of (Np,d), states increment for each particle
                dU: numpy array with the shape of (Np,d), optimal control input for each particle
        """
        dZ = np.reshape(Y*self.dt, [1, Y.shape[0]])
        dI = dZ-0.5*(self.particles.h+self.h_hat)*self.dt
        # dU = np.zeros(self.particles.X.shape)
        # dU[:,0] = Y*self.dt
        dX = self.f(self.particles.X)*self.dt + self.sigma_V*np.random.normal(0, np.sqrt(self.dt), size=self.particles.X.shape) # +dU
        dU = self.optimal_control(dI, dZ)
        # print('dX=',np.mean(dX,axis=0),'dU=',np.mean(dU,axis=0))
        self.particles.update(dX+dU)
        self.h_hat = self.calculate_h_hat()
        return dU

    def run(self, Y, show_time=False):
        """Run FPF with time series observations(Y)"""
        h_hat = np.zeros(Y.shape)
        h = np.zeros([self.particles.Np, self.m, Y.shape[1]])
        X = np.zeros([self.particles.Np, self.d, Y.shape[1]])
        dU = np.zeros([self.particles.Np, self.d, Y.shape[1]])
        c = np.zeros([self.m, self.galerkin.L, Y.shape[1]])
        for k in range(Y.shape[1]):
            if show_time and np.mod(k,int(1/self.dt))==0:
                print("===========time: {} [s]==============".format(int(k*self.dt)))
            
            h_hat[:,k] = self.h_hat
            h[:,:,k] = self.particles.h
            X[:,:,k] = self.particles.X
            dU[:,:,k] = self.update(Y[:,k])
            c[:,:,k] = self.galerkin.c

        return Struct(states_label=self.states_label, h_hat=h_hat, h=h, X=X, c=c)

class Figure(object):

    def __init__(self, fig_property, Y, filtered_signal):
        self.fig_property = fig_property
        self.Y = Y
        self.filtered_signal = filtered_signal
        self.figs = dict()

        # set properties for plot_X and histogram if needed
        if (('plot_X' in self.fig_property.__dict__) and self.fig_property.plot_X) \
            or (('plot_X_quantile' in self.fig_property.__dict__) and self.fig_property.plot_X_quantile):
            # set time skip 
            if not 't_skip' in self.fig_property.__dict__:
                self.fig_property.t_skip = 1

            # set sampled index
            Np = filtered_signal.X.shape[0]
            if 'particles_ratio' in self.fig_property.__dict__ :
                self.sampled_index = np.random.choice(Np, size=int(Np*self.fig_property.particles_ratio), replace=False)
            else :
                self.sampled_index = np.array(range(Np)) if Np<10 else np.random.choice(Np, size=10, replace=False)

            # set histogram property
            if not 'histogram' in self.fig_property.__dict__:
                self.fig_property.histogram = Struct(t=['-1'], n_bins=100)
            else:
                if not 'n_bins' in self.fig_property.histogram.__dict__:
                    self.fig_property.histogram = Struct(t=self.fig_property.histogram.t, n_bins=100)
                if not 't' in self.fig_property.histogram.__dict__:
                    self.fig_property.histogram = Struct(t=['-1'], n_bins=self.fig_property.histogram.n_bins)

            if len(self.fig_property.histogram.t) == 0:
                self.fig_property.histogram = Struct(length=0, n_bins=self.fig_property.histogram.n_bins)
            else:
                t_index = [None] * len(self.fig_property.histogram.t)
                for k, t in enumerate(self.fig_property.histogram.t):
                    t_index[k] = np.mod(Y.t.shape[0]+int(t), Y.t.shape[0]) if isinstance(t, str) \
                        else find_closest(t, Y.t.tolist(), pos=True)
                t_index = list(dict.fromkeys(t_index))
                t_index.sort()
                self.fig_property.histogram = Struct(
                    t = Y.t[t_index], 
                    t_index = t_index, 
                    length = len(t_index), 
                    n_bins = self.fig_property.histogram.n_bins
                )

        if (('plot_X_quantile' in self.fig_property.__dict__) and self.fig_property.plot_X_quantile):
            if not 'mean_on_circle' in self.fig_property.__dict__:
                self.fig_property.mean_on_circle = [False] * len(self.filtered_signal.states_label)
            pass

        return

    def plot_figures(self, show=False):
        if ('plot_signal' in self.fig_property.__dict__) and self.fig_property.plot_signal:
            self.figs['signal'] = self.plot_signal()
        if ('plot_X' in self.fig_property.__dict__) and self.fig_property.plot_X:
            self.figs['X'] = self.plot_X()
        if ('plot_X_quantile' in self.fig_property.__dict__) and self.fig_property.plot_X_quantile:
            self.figs['X_quantile'] = self.plot_X_quantile()
        if ('plot_c' in self.fig_property.__dict__) and self.fig_property.plot_c:
            self.figs['c'] = self.plot_c()
        if show:
            self.show()
        return self.figs
    
    def plot_signal(self):
        print("plot_signal")
        specs = [ [{'rowspan':3}] * self.Y.value.shape[0],
                [None] * self.Y.value.shape[0],
                [None] * self.Y.value.shape[0],
                [{}] * self.Y.value.shape[0] ]
        fig = subplots.make_subplots(
            rows = len(specs), 
            cols = self.Y.value.shape[0],
            specs = specs,
            shared_xaxes = True
        )

        for m in range(self.Y.value.shape[0]):
            fig.append_trace(
                trace=go.Scatter(
                    x = self.Y.t,
                    y = self.Y.value[m,:],
                    name = 'signal',
                    line = dict(color='grey')
                ), row=1, col=m+1)
            fig.append_trace(
                trace=go.Scatter(
                    x = self.Y.t,
                    y = self.filtered_signal.h_hat[m,:],
                    name = 'estimation',
                    line = dict(color='magenta')
                ), row=1, col=m+1)
            fig.append_trace(
                trace=go.Scatter(
                    x = self.Y.t,
                    y = self.filtered_signal.h_hat[m,:] - self.Y.value[m,:],
                    name = 'error',
                    line = dict(color='magenta'),
                    showlegend = False
                ), row=len(specs), col=m+1)
        
        error_yaxis = {'yaxis{}'.format(self.Y.value.shape[0]+1):dict(title='error')}
        fig['layout'].update(
            showlegend = True,
            font = dict(size=self.fig_property.fontsize),
            xaxis = dict(title='time [s]'),
            **error_yaxis
        )
        return fig

    def plot_X(self):
        print("plot_X")
        if 'get_states' in self.fig_property.__dict__:
            self.filtered_signal.X = self.fig_property.get_states(self.filtered_signal.X)
        
        fig =  self.states_figure()
            
        for d in range(self.filtered_signal.X.shape[1]):
            for color_index, p in enumerate(self.sampled_index):
                fig.append_trace(
                    trace=go.Scatter(
                        x = self.Y.t[::self.fig_property.t_skip],
                        y = self.filtered_signal.X[p,d,::self.fig_property.t_skip],
                        mode = 'markers',
                        name = '$X^{}$'.format(p),
                        line = dict(color=default_colors[np.mod(color_index,len(default_colors))]),
                        legendgroup = 'X^{}'.format(p),
                        showlegend = True #if d==0 else False
                    ), row=d+1, col=1)
        
            for k in range(self.fig_property.histogram.length):
                fig.append_trace(trace=go.Histogram(
                    y = self.filtered_signal.X[:,d,self.fig_property.histogram.t_index[k]],
                    histnorm = 'probability',
                    nbinsy = self.fig_property.histogram.n_bins,
                    marker=dict(color='rgb(50, 50, 125)'),
                    name = 't={}'.format(self.fig_property.histogram.t[k]),
                    showlegend = False
                ), row=d+1, col=self.fig_property.histogram.length+k+1)

        return fig

    def plot_X_quantile(self):
        print("plot_X_quantile")
        q1 = np.quantile(self.filtered_signal.X, 0.25, axis=0)
        mean = np.mean(self.filtered_signal.X, axis=0)
        q3 = np.quantile(self.filtered_signal.X, 0.75, axis=0)
        q1 -= mean
        q3 -= mean
        if 'get_states' in self.fig_property.__dict__:
            mean = self.fig_property.get_states(mean)
            self.filtered_signal.X = self.fig_property.get_states(self.filtered_signal.X)
        q1 += mean
        q3 += mean
        
        for d in range(len(self.filtered_signal.states_label)):
            if self.fig_property.mean_on_circle[d]:
                mean[d,:], l = circle_mean(self.filtered_signal.X[:,d,:])
                q1[d,:] = mean[d,:] - np.exp(-l)
                q3[d,:] = mean[d,:] + np.exp(-l)
                print(l, l.shape)

        fig =  self.states_figure()
        for d in range(self.filtered_signal.X.shape[1]):
            fig.append_trace(trace=go.Scatter(
                    x = self.Y.t,
                    y = mean[d,:],
                    name = 'E[$X_{}$]'.format(d+1),
                    line = dict(color='magenta')
                ), row=d+1, col=1)
            fig.add_trace(trace=go.Scatter(
                x = self.Y.t, 
                y = q1[d],
                fill = None,
                mode = 'lines',
                line_color = 'grey',
                ), row=d+1, col=1)
            fig.add_trace(trace=go.Scatter(
                x = self.Y.t, 
                y = q3[d],
                fill = 'tonexty',
                mode = 'lines',
                line_color = 'grey',
                ), row=d+1, col=1)

        
            for k in range(self.fig_property.histogram.length):
                fig.append_trace(trace=go.Histogram(
                    y = self.filtered_signal.X[:,d,self.fig_property.histogram.t_index[k]],
                    histnorm = 'probability',
                    nbinsy = self.fig_property.histogram.n_bins,
                    marker=dict(color='rgb(50, 50, 125)'),
                    name = 't={}'.format(self.fig_property.histogram.t[k]),
                    showlegend = False
                ), row=d+1, col=self.fig_property.histogram.length+k+1)

        return fig

    def states_figure(self):
        if self.fig_property.histogram.length==0 :
            row_spec = [{}]
        else:
            row_spec = [{'colspan':self.fig_property.histogram.length}]
            row_spec.extend([None] * (self.fig_property.histogram.length-1))
            row_spec.extend([{}] * self.fig_property.histogram.length)
        specs = [row_spec] * self.filtered_signal.X.shape[1]

        fig = subplots.make_subplots(
            rows = self.filtered_signal.X.shape[1],
            cols = max(2*self.fig_property.histogram.length,1),
            specs = specs,
            shared_xaxes = True,
            shared_yaxes = True
        )

        limits = np.zeros((self.filtered_signal.X.shape[1],2))
        for d in range(self.filtered_signal.X.shape[1]):
            limits[d,:] = find_limits(self.filtered_signal.X[:,d,:])

        xaxis = dict()
        t_lines = []
        for k in range(self.fig_property.histogram.length):
            xaxis['xaxis{}'.format(self.fig_property.histogram.length+k+1)] = dict(title='t={}'.format(self.fig_property.histogram.t[k]), range=[0,1], dtick=0.25)
            for d in range(self.filtered_signal.X.shape[1]):
                t_lines.append(dict(
                    type = 'line', 
                    xref = 'x1',
                    yref = 'y{}'.format(d+1),
                    x0 = self.fig_property.histogram.t[k],
                    x1 = self.fig_property.histogram.t[k],
                    y0 = limits[d,0],
                    y1 = limits[d,1],
                    line = dict(color='rgb(50, 50, 125)', width=3)
                ))
        yaxis = dict()
        for d in range(self.filtered_signal.X.shape[1]):
            yaxis['yaxis{}'.format(d+1)] = dict(title=self.filtered_signal.states_label[d], range=limits[d,:])
        fig['layout'].update(
            font = dict(size=self.fig_property.fontsize),
            showlegend = False,
            shapes = t_lines,
            xaxis1 = dict(title='time [s]'),
            **xaxis,
            **yaxis
        )

        return fig

    def plot_c(self):
        print("plot_c")

        fig = subplots.make_subplots(
            rows = self.filtered_signal.c.shape[1],
            cols = self.filtered_signal.c.shape[0],
            shared_xaxes = True,
            shared_yaxes = False
        )

        for m in range(self.filtered_signal.c.shape[0]):
            for l in range(self.filtered_signal.c.shape[1]):
                fig.append_trace(
                    trace=go.Scatter(
                        x = self.Y.t,
                        y = self.filtered_signal.c[m,l,:],
                        name = r'$\kappa{}{}$'.format(m+1,l+1),
                        line = dict(color='magenta'),
                    ), row=l+1, col=m+1)
        
        yaxis = dict()
        for l in range(self.filtered_signal.c.shape[1]):
            yaxis['yaxis{}'.format(self.filtered_signal.c.shape[0]*l+1)] = dict(title=r'$\kappa_{}$'.format(l+1))

        fig['layout'].update(
            showlegend = False,
            font = dict(size=self.fig_property.fontsize),
            xaxis = dict(title='time [s]'),
            **yaxis
        )

        return fig
    
    def show(self):
        for name, fig in self.figs.items():
            pyo.plot(fig, filename=name+'.html', include_mathjax='cdn')

class Model_SinusoidalWave(Model):
    def __init__(self, freq_range, amp_range, sigma_V, sigma_W, sigma_V_min=0.01, sigma_W_min=0.1):
        self.freq_min=freq_range[0]
        self.freq_max=freq_range[1]
        self.sigma_V_min = sigma_V_min
        self.sigma_W_min = sigma_W_min
        Model.__init__(self, m=1, d=2, X0_range=[[0,2*np.pi],amp_range], 
                        states=['theta','r'], states_label=[r'$\theta$',r'$r$'], 
                        sigma_V=self.modified_sigma_V(sigma_V), 
                        sigma_W=self.modified_sigma_W(sigma_W))

    def modified_sigma_V(self, sigma_V):
        sigma_V = np.array(sigma_V).clip(min=self.sigma_V_min)
        sigma_V[0::2] = 0
        return sigma_V
    
    def modified_sigma_W(self, sigma_W):
        sigma_W = np.array(sigma_W).clip(min=self.sigma_W_min)
        return sigma_W

    def states_constraints(self, X):
        return X

    def get_states(self, X):
        if X.ndim == 1:
            X[0] = np.mod(X[0], 2*np.pi)
        if X.ndim == 2:
            X[0,:] = np.mod(X[0,:], 2*np.pi)
        if X.ndim == 3:
            X[:,0,:] = np.mod(X[:,0,:], 2*np.pi)
        return X

    def f(self, X):
        dXdt = np.zeros(X.shape)
        dXdt[:,0] = np.linspace(self.freq_min*2*np.pi, self.freq_max*2*np.pi, X.shape[0])
        return dXdt

    def h(self, X):
        return X[:,1]*np.cos(X[:,0]) 

class Galerkin_SinusoidalWave(Galerkin):    # 2 states (theta, r) and 2 base functions [r*cos(theta), r*sin(theta)]
    def __init__(self, states, m):
        Galerkin.__init__(self, states, m, L=2)
        
    def psi(self, X):
        trial_functions = [ X[:,1]*np.cos(X[:,0]), X[:,1]*np.sin(X[:,0])]
        return np.array(trial_functions)
        
    def grad_psi(self, X):
        grad_trial_functions = [[ -X[:,1]*np.sin(X[:,0]), np.cos(X[:,0])],\
                                [  X[:,1]*np.cos(X[:,0]), np.sin(X[:,0])]]
        return np.array(grad_trial_functions)

    def grad_grad_psi(self, X):
        grad_grad_trial_functions = [[[ -X[:,1]*np.cos(X[:,0]),           -np.sin(X[:,0])],\
                                      [        -np.sin(X[:,0]), np.zeros(X[:,0].shape[0])]],\
                                     [[ -X[:,1]*np.sin(X[:,0]),            np.cos(X[:,0])],\
                                      [         np.cos(X[:,0]), np.zeros(X[:,0].shape[0])]]]
        return np.array(grad_grad_trial_functions)
        
if __name__ == "__main__":
    
    T = 5.5
    sampling_rate = 100 # Hz
    dt = 1./sampling_rate

    sigma_V = [0.1]
    signal_type = Sinusoids(dt, amp=[[1]], freq=[1], theta0=[[np.pi/2]], sigma_V=sigma_V, SNR=[20])
    Y, _ = Signal.create(signal_type=signal_type, T=T)
    
    Np = 1000
    model = Model_SinusoidalWave(freq_range=[0.9,1.1], amp_range=[0.9,1.1], sigma_V=sigma_V, sigma_W=signal_type.sigma_W)
    galerkin = Galerkin_SinusoidalWave(model.states, model.m)
    fpf = FPF(number_of_particles=Np, model=model, galerkin=galerkin, dt=signal_type.dt)
    filtered_signal = fpf.run(Y.value)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, plot_signal=True, plot_X=True, plot_c=True)
    figure = Figure(fig_property=fig_property, Y=Y, filtered_signal=filtered_signal)
    figure.plot_figures(show=True)
    pass


