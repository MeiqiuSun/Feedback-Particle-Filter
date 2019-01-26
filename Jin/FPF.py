# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:05:59 2016

@author: Jinwon Kim
"""
from numpy import cos, sin, mean, sqrt, zeros, ones, arctan2, copy, array, log
from numpy import dot, sum, exp, einsum, vstack
from numpy.linalg import solve, norm
from numpy.random import normal
#from scipy.optimize import fixed_point
from PyQt5 import QtGui,QtCore,Qt,QtWidgets
import pyqtgraph as pg

class FPF_phase(object):
    """
    FPF_phase: particles has only 1D state which denotes phase in limit cycle.
        Initialize = FPF_obj(X0,M,g,h,sW,sB,dt,harg,adapt)
            X0: numpy array(N), initial value of X, define N
            M: integer, dimension of y or dZ
            g: function, g(N) will return nparray of N frequencies
            h: function, h(X,harg) = numpy array(M,len(X))
            sW, sB: float, size of noise, sW cannot be zero
            dt: float, size of time step
            harg: any array, varying parameter of h for adaptaion
            adapt: function, harg += adapt(X,y,h_hat)*dt
            scale: float, scales the gain of FPF, e.g. setting sc = 0.01 has same effect of sW x10
        Methods =
            run_step_y(y): calculate K, update X and h, then update harg(if adapt)
            run_step_dZ(dZ): run_step_y(dZ/dt)
            get_X_hat(): calculate mean phase of particles
            get_R(): calculate order parameter R
        Useful Members = 
            X: numpy array(N), phase of particles
            y: input reference signal
            h: numpy array(M,N), h(Xi)
            h_hat: numpy array(M): mean value of h(Xi)
    """
    def __init__(self, X0, M, g, h, sW, sB, dt, harg=None, adapt=False, scale=1.0, learn_sigW=False):
        #fundamental parameters
        self.N = len(X0)
        self.M = M
        self.X = copy(X0)
        
        self.ws = g(self.N)
        self.w = mean(self.ws)
        
        self.sB = copy(sB)
        self.sW2 = (sW ** 2) * ones(self.M)
        self.alpha = 0.6
        
        #measurement parameters
        self.dt = copy(dt)
        self.y = zeros(self.M)
        
        self.scale = copy(scale)
        
        self.eval_h = h
        self.h_hat = zeros(self.M)
        self.harg = copy(harg)
        self.adapt = adapt
        
        self.ls = learn_sigW
        
        self.h = self.eval_h(self.X,self.harg)
        self.h_hat = mean(self.h,axis=1)
            
    def run_step_dZ(self, dZ):
        self.run_step_y(dZ/self.dt)
        
    def run_step_y(self, y):
        self.y = y
        if self.adapt: self.harg += self.adapt(self.X,self.y,self.h_hat,self.harg,self.w)*self.dt
        A = self.calc_A()
        dX = zeros(self.N)
        if self.ls:
            dIsW = (y-self.h_hat)*self.dt
            self.sW2 = (1-self.alpha*self.dt)*self.sW2 + self.alpha*dIsW*dIsW
        for j in range(self.M):
            K, Kp  = self.calc_K(A,self.h[j]-self.h_hat[j])
            dI = y[j] - 0.5*(self.h[j] + self.h_hat[j]) #actually, this is dI/dt
            dX += K*(dI + 0.5*Kp)*self.dt/self.sW2[j]*self.scale
        self.X += self.ws*self.dt + self.sB*sqrt(self.dt)*normal(0,1,self.N) + dX #% (2*np.pi)
        self.h = self.eval_h(self.X,self.harg)
        self.h_hat = mean(self.h,axis=1)
            
    def get_X_hat(self):
        am = arctan2(mean(sin(self.X)),mean(cos(self.X)))
        #if am < 0: am += 2*pi
        return am
        
    def get_X(self):
        return self.X
        
    def get_R(self):
        return sqrt(mean(cos(self.X)) ** 2 + mean(sin(self.X)) ** 2)
    
    def calc_K(self,A,H): #H = hi - h_hat
        b = zeros(2)
        b[0] = mean(sin(self.X)*H)
        b[1] = mean(-cos(self.X)*H)
        k = solve(A,b)
        K = k[0]*cos(self.X) + k[1]*sin(self.X)
        Kp = k[1]*cos(self.X) - k[0]*sin(self.X)
        return K, Kp
    
    def calc_A(self):
        A = zeros([2,2])
        c = cos(self.X)
        s = sin(self.X)
        A[0,0] = mean(c*c)
        A[0,1] = mean(s*c)
        A[1,0] = A[0,1]
        A[1,1] = mean(s*s)
        return A

'============================================================================='

class FPF_phase_K(FPF_phase):
    """
    FPF_phase_K: FPF_phase with kernel based gain function approximation method
    """        
    def run_step_y(self, y):
        self.y = y
        if self.adapt: self.harg += self.adapt(self.th,self.y,self.h_hat,self.w)*self.dt
        dth = zeros(self.N)
        if self.ls:
            dIsW = (y-self.h_hat)*self.dt
            self.sW2 = (1-self.alpha*self.dt)*self.sW2 + self.alpha*dIsW*dIsW
        for j in range(self.M):
            phi,K,KKp  = self.calc_gain(self.X,self.h[j]-self.h_hat[j],self.phi[j],self.eps)
            self.phi[j] = phi
            dI = y[j] - 0.5*(self.h[j] + self.h_hat[j]) #actually, this is dI/dt
            dth += (K*dI + 0.5*KKp)*self.dt/self.sW2[j]*self.scale
        self.th += self.ws*self.dt + self.sB*sqrt(self.dt)*normal(0,1,self.N) + dth #% (2*np.pi)
        self.X = vstack((cos(self.th),sin(self.th)))
        self.h = self.eval_h(self.th,self.harg)
        self.h_hat = mean(self.h,axis=1)
    
    def run_step_dZ(self, dZ):
        self.run_step_y(dZ/self.dt)
        
    def calc_gain(self,X,H,phi0,e):
        T = self.calc_T(X,e)
        eH = e*H
        phi = self.calc_phi(T,eH,phi0)#fixed_point(lambda phi:dot(T,phi)+eH,phi0,maxiter=20000,method='iteration')
        phi -= mean(phi)
        K = zeros(X.shape)
        for d in range(self.d):
            K[d] = (dot(T,X[d]*phi)-dot(T,phi)*dot(T,X[d]))/(2*e)+(dot(T,X[d]*H)-dot(T,H)*dot(T,X[d]))/2
        Hess_phi = self.eval_Hess_phi(X,T,phi+e*H)/(4*e*e)
        KKp = einsum('ijk,jk->ik',Hess_phi,K)
        
        Kth = X[0]*K[1]-X[1]*K[0]
        KKpth = X[0]*KKp[1]-X[1]*KKp[0]
        #return phi,K,KKp 
        return phi,Kth,KKpth
        
    def calc_T(self,X,e):
        ge = zeros([self.N,self.N])
        ke = zeros([self.N,self.N])
        for i in range(self.N):
            for j in range(i,self.N):
                dXij = X[:,i]-X[:,j]
                ge[i,j] = exp(-dot(dXij,dXij)/(4*e))
                ge[j,i] = ge[i,j]
        sumge = sum(ge,axis=1)
        for i in range(self.N):
            for j in range(i,self.N):
                ke[i,j] = ge[i,j]/sqrt(sumge[i]*sumge[j])
                ke[j,i] = ke[i,j]        
        sumke = sum(ke,axis=1)
        return ke/sumke
    
    def calc_phi(self,T,eH,phi0):
        p0 = phi0
        for i in range(10000):
            p1 = dot(T,p0)+eH
            p1 -= mean(p1)
            if norm(p1-p0) < 0.000001: return p1
            p0 = p1
        return p1
        
    def eval_Hess_phi(self,X,T,phieH):  #H(phi) -> compute K'
        Hess_phi = zeros([self.d,self.d,self.N])
        for i in range(self.d):
            for j in range(i,self.d):
                Hess_phi[i,j] += dot(T,X[i]*X[j]*phieH)
                Hess_phi[i,j] -= dot(T,X[i]*phieH)*dot(T,X[j])
                Hess_phi[i,j] -= dot(T,X[j]*phieH)*dot(T,X[i])
                Hess_phi[i,j] -= dot(T,X[i]*X[j])*dot(T,phieH)
                Hess_phi[i,j] += 2*dot(T,X[i])*dot(T,X[j])*dot(T,phieH)
                Hess_phi[i,j]  = Hess_phi[i,j]
                Hess_phi[j,i]  = Hess_phi[i,j]    
        return Hess_phi        
            
class SC_GCF(object):
    """
    SC_GCF: particles has only 1D state which denotes phase in limit cycle.
        K calculated from Galerkin method
        clusturing matrix E is considered. ideally, Eij = 1 if Xi and Xj are synchronized, 0 if not
        rho[i] is a rotation number of each particles
        rho[i] = w[i] at t=0, and developed by dr = -b(r*dt - dth)
        E = exp(-l*(rho[j]-rho[i])^2), l chosed to be log(2)/2Dw, where Dw is frequency difference between particles
        hh_weighted is considered. hh_wtd[i] = mean(h[j]*exp(-l*(rho[j]-rho[i])^2))
        dX[i] = w[i]dt + K*(y-0.5*(h[i]+hh_wtd[i]))
    """
    def __init__(self, X0, M, g, h, sW, sB, dt, harg=None, adapt=False, scale=1.0, learn_sigW=False):
        #fundamental parameters
        self.N = len(X0)
        self.M = M
        self.X = copy(X0)
        
        self.ws = g(self.N)
        self.ws.sort()
        self.w = mean(self.ws)
        
        self.sB = copy(sB)
        self.sW2 = (sW ** 2) * ones(self.M)
        self.a = 0.6
        
        #neighborhood properties
        self.z = 0.1
        Dw = max(self.ws)-min(self.ws)
        #self.l = log(2)/(2*Dw/self.N)**2
        self.l = log(2)/(Dw/20)**2
        self.rho = copy(self.ws)
        self.K = 1.5
        
        #measurement parameters
        self.dt = copy(dt)
        self.y = zeros(self.M)
        
        self.scale = copy(scale)
        
        self.eval_h = h
        #self.h_hat = zeros(self.M)
        self.harg = copy(harg)
        self.adapt = adapt
        
        self.ls = learn_sigW
        
        self.h = self.eval_h(self.X,self.harg)
        E = self.calc_E(self.rho)
        sum_E = sum(E,axis=1)
        self.hh_wtd = self.calc_hh_wtd(self.h,E,sum_E)
        self.h_hat = array([sum(self.hh_wtd[j]/sum_E) for j in range(self.M)])
            
    def run_step_dZ(self, dZ):
        self.run_step_y(dZ/self.dt)
        
    def run_step_y(self, y):
        self.y = y
        E = self.calc_E(self.rho)
        sum_E = sum(E,axis=1)
        if self.adapt: self.harg += self.adapt(self.X,self.y,self.h_hat,self.harg,E,sum_E)*self.dt
        A = self.calc_A(E,sum_E)
        dX = zeros(self.N)
        if self.ls:
            dIsW = (y-self.h_hat)*self.dt
            self.sW2 += self.a*(dIsW*dIsW-self.sW2*self.dt)
        for j in range(self.M):
            K, Kp  = self.calc_K(A,self.h[j]-self.hh_wtd[j],E,sum_E)
            dI = y[j] - self.h_hat[j] - 0.5*(self.h[j] - self.hh_wtd[j]) #actually, this is dI/dt
            dX += K*(dI + 0.5*Kp)*self.dt/self.sW2[j]*self.scale
        dX += self.ws*self.dt + self.sB*sqrt(self.dt)*normal(0,1,self.N)
        self.rho += self.z*(dX - self.rho*self.dt)
        self.X += dX #% (2*np.pi)
        self.h = self.eval_h(self.X,self.harg)
        self.hh_wtd = self.calc_hh_wtd(self.h,E,sum_E)
        self.h_hat = array([sum(self.hh_wtd[j]/sum_E) for j in range(self.M)])
            
    def get_X_hat(self):
        am = arctan2(mean(sin(self.X)),mean(cos(self.X)))
        #if am < 0: am += 2*pi
        return am
        
    def get_X(self):
        return self.X
        
    def get_R(self):
        return sqrt(mean(cos(self.X)) ** 2 + mean(sin(self.X)) ** 2)
    
    def calc_K(self,A,H,E,sum_E): #H = hi - h_hat
        b = zeros([2,self.N])
        b[0] = dot(E,sin(self.X)*H)/sum_E#self.N TODO:verify it is right to use sumE
        b[1] = dot(E,-cos(self.X)*H)/sum_E#self.N
        k = zeros([2,self.N])
        for i in range(self.N):
            k[:,i] = [0,0] if sum_E[i] == 1.0 else solve(A[:,:,i],b[:,i])
        K = k[0]*cos(self.X) + k[1]*sin(self.X)
        Kp = k[1]*cos(self.X) - k[0]*sin(self.X)
        return K, Kp
    
    def calc_A(self,E,sum_E):
        A = zeros([2,2,self.N])
        c = cos(self.X)
        s = sin(self.X)
        A[0,0] = dot(E,c*c)/sum_E
        A[0,1] = dot(E,s*c)/sum_E
        A[1,0] = A[0,1]
        A[1,1] = dot(E,s*s)/sum_E
        return A
        
    def calc_E(self,rho):
        E = zeros([self.N,self.N])
        for i in range(self.N):
            for j in range(i,self.N):
                E[i,j] = exp(-self.l*(rho[i]-rho[j])**2)
                E[j,i] = E[i,j]
        return E
        
    def calc_hh_wtd(self,h,E,sum_E):
        hh_wtd = zeros([self.M,self.N])
        for j in range(self.M):
            hh_wtd[j] = dot(E,h[j])/sum_E
        return hh_wtd
        
'============================================================================='
        
class Qt_FPF_Plot(QtWidgets.QMainWindow):
    def __init__(self, FPF, plt_index = 0):
        super(Qt_FPF_Plot,self).__init__()
        self.resize(850,450)
        self.move(400,300)
        self.setWindowTitle('FPF particles')
        
        self.FPF = FPF
        self.N = FPF.N
        self.th = zeros(self.N)
        self.xmean = 1.0
        self.ymean = 0.0
        
        self.set_fig_params()
        self.create_fig()
        self.startTimer(int(self.FPF.dt*500))
        self.show()

        self.plt_index = plt_index
        self.ts = []
        self.ys = []
        self.hs = []
        self.Rs = []
        self.Nt = 500
    
    def set_fig_params(self):
        self.colorb = QtGui.QColor(0,70,170)
        self.coloro = QtGui.QColor(170,70,0)
        self.penb = pg.mkPen (color = (0,70,170),width = 1.5 )
        self.peno = pg.mkPen (color = (170,70,0),width = 1 )
        self.rb = 5
        self.rB = 200
        self.xc = 625
        self.yc = 225
        
    def append_data(self,t):
        self.ts.append(t)
        self.ys.append(copy(self.FPF.y[self.plt_index]))
        self.hs.append(copy(self.FPF.h_hat[self.plt_index]))
        self.Rs.append(self.FPF.get_R())
        if len(self.ts)>self.Nt:
            self.ts.pop(0)
            self.ys.pop(0)
            self.hs.pop(0)
            self.Rs.pop(0)

    def create_fig(self):
        self.figyh = pg.PlotWidget(self)
        self.figyh.move(25,25)
        self.figyh.resize(375,188)
        self.figyh.setYRange(-3,3)
        self.liney = self.figyh.plot(pen=self.peno)
        self.lineh = self.figyh.plot(pen=self.penb)
        
        self.figR = pg.PlotWidget(self)
        self.figR.move(25,237)
        self.figR.resize(375,188)
        self.figR.setYRange(0,1)
        self.lineR = self.figR.plot(pen=self.penb)
    
    def update_fig(self):
        Ntt = len(self.ts)
        self.liney.setData(self.ts[:Ntt],array(self.ys)[:Ntt])
        self.lineh.setData(self.ts[:Ntt],array(self.hs)[:Ntt])
        self.lineR.setData(self.ts[:Ntt],array(self.Rs)[:Ntt])
            
    def timerEvent(self,e):
        self.update_fig()
        self.get_FPF()
        self.update()
    
    def get_FPF(self):        
        self.th = self.FPF.get_X()
        self.xmean = mean(cos(self.th))
        self.ymean = mean(sin(self.th))
        
    def paintEvent(self,e):
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.drawEllipse(self.xc-self.rB,self.yc-self.rB,2*self.rB,2*self.rB)
        qp.setPen(self.colorb)
        qp.setBrush(self.colorb)
        for th in self.th:
            qp.drawEllipse(self.xc + self.rB*cos(th)-self.rb/2,self.yc + self.rB*sin(th)-self.rb/2,self.rb,self.rb)
        qp.setPen(self.peno)
        qp.setBrush(self.coloro)
        psix = self.xc + self.rB*self.xmean
        psiy = self.yc + self.rB*self.ymean
        qp.drawLine(self.xc,self.yc,psix,psiy) 
        qp.drawEllipse(psix - self.rb/2,psiy - self.rb/2,self.rb,self.rb)
        qp.end()
        
'============================================================================='
        
class FPF_aug_c1(object):
    """
    FPF_augmented_harg: particles has 2D state, phase in limit cycle and 1 coeff for h.
        Initialize = FPF_obj(X0,harg0,M,g,h,sW,sB,dt)
            X0: numpy array(N), initial value of X, define N
            harg0: numpy array(N), initial value of harg,
            M: integer, dimension of y or dZ
            g: function, g(N) will return nparray of N frequencies
            h: function, h(X) = numpy array(M,N)
            sW, sB: float, size of noise, sW cannot be zero
            dt: float, size of time step
            scale: float, scaling factor the gain of FPF
        Methods =
            run_step_y(y): calculate K, update X and h, then update harg(if adapt)
            run_step_dZ(dZ): run_step_y(dZ/dt)
            get_X_hat(): calculate mean phase of particles
            get_R(): calculate order parameter R
        Useful Members = 
            X: numpy array(N), phase of particles
            y: input reference signal
            h: numpy array(M,N), h(Xi)
            h_hat: numpy array(M): mean value of h(Xi)
        Note = 
            Galerkin Aprroximation using basis function {sinX1, -cosX1, X2, 0.5*X2^2}
    """
    def __init__(self, X0, harg0, M, g, h, sWsig, sB, dt, scale=1.0):
        #fundamental parameters
        self.N = len(X0)
        self.M = M
        self.X = vstack((copy(X0),copy(harg0)))
        
        self.ws = g(self.N)
        self.w = mean(self.ws)
        self.ws = vstack((self.ws,zeros(self.N)))#vstack((self.ws,zeros(h0.shape)))
        
        self.sB = copy(sB)
        self.sW2 = sWsig ** 2
        
        #measurement parameters
        self.dt = copy(dt)
        self.y = zeros(self.M)
        
        self.scale = copy(scale)
        
        self.eval_h = h
        self.h_hat = zeros(self.M)
        
        self.h = self.eval_h(self.X)
        self.h_hat = mean(self.h,axis=1)
            
    def run_step_dZ(self, dZ):
        self.run_step_y(dZ/self.dt)
        
    def run_step_y(self, y):
        self.y = y
        #A1, A2 = self.calc_A()
        A = self.calc_A()
        dX = zeros((2, self.N))
        for j in range(self.M):
            K, Kp,  = self.calc_K(A,self.h[j]-self.h_hat[j])
            #K, Kp,  = self.calc_K(A1,A2,self.h[j],self.h_hat[j])
            dI = y[j] - 0.5*(self.h[j] + self.h_hat[j]) #actually, this is dI/dt
            dX += K*(dI + 0.5*Kp)*self.dt/self.sW2
        self.X += self.ws*self.dt + vstack((self.sB*sqrt(self.dt)*normal(0,1,self.N),zeros(self.N))) + dX
        self.h = self.eval_h(self.X)
        self.h_hat = mean(self.h,axis=1)
            
    def get_X_hat(self):
        am = arctan2(mean(sin(self.X[0])),mean(cos(self.X[0])))
        return am
        
    def get_X(self):
        return self.X[0]
        
    def get_R(self):
        return sqrt(mean(cos(self.X[0])) ** 2 + mean(sin(self.X[0])) ** 2)
    
    def calc_K(self,A,H): 
        b = zeros(3)
        b[0] = mean(self.X[1]*sin(self.X[0])*H)
        b[1] = mean(-self.X[1]*cos(self.X[0])*H)
        b[2] = mean(self.X[1]*H)
        k = solve(A,b)*self.scale
        K1 = k[0]*self.X[1]*cos(self.X[0]) + k[1]*self.X[1]*sin(self.X[0])
        Kp1 = k[1]*self.X[1]*cos(self.X[0]) - k[0]*self.X[1]*sin(self.X[0])
        K2 = k[0]*sin(self.X[0]) - k[1]*cos(self.X[0]) + k[2]
        Kp2 = zeros(self.N)
        return vstack((K1,K2)), vstack((Kp1,Kp2))
    
    def calc_A(self):
        A = zeros([3,3])
        dp11 = self.X[1]*cos(self.X[0])
        dp21 = self.X[1]*sin(self.X[0])
        dp12 = sin(self.X[0])
        dp22 = -cos(self.X[0])
        dp31 = zeros(self.N)
        dp32 = ones(self.N)
        A[0,0] = mean(dp11*dp11 + dp12*dp12)
        A[0,1] = mean(dp11*dp21 + dp12*dp22)
        A[0,2] = mean(dp11*dp31 + dp12*dp32)
        
        A[1,0] = A[0,1]#mean(dp21*dp11 + dp22*dp12)
        A[1,1] = mean(dp21*dp21 + dp22*dp22)
        A[1,2] = mean(dp21*dp31 + dp22*dp32)
        
        A[2,0] = A[0,2]#mean(dp31*dp11 + dp32*dp12)
        A[2,1] = A[1,2]#mean(dp31*dp21 + dp32*dp22)
        A[2,2] = mean(dp31*dp31 + dp32*dp32)
        return A

#    def calc_K(self,A1,A2,h,h_hat):
#        b = zeros(2)
#        b[0] = mean(sin(self.X[0])*(h - h_hat))
#        b[1] = mean(-cos(self.X[0])*(h - h_hat))
#        k = solve(A1,b)*self.sW2inv*self.scale
#        K1 = k[0]*cos(self.X[0]) + k[1]*sin(self.X[0])
#        Kp1 = k[1]*cos(self.X[0]) - k[0]*sin(self.X[0])
#        b[0] = mean(self.X[1]*(h - h_hat))
#        b[1] = mean(0.5*self.X[1]*self.X[1]*(h - h_hat))
#        k = solve(A2,b)*self.sW2inv*self.scale
#        K2 = k[0] + k[1]*self.X[1]
#        Kp2 = k[1]*ones(self.N)
#        return vstack((K1,K2)), vstack((Kp1,Kp2))
#    
#    def calc_A(self):
#        A1 = zeros([2,2])
#        A2 = zeros([2,2])
#        dp11 = cos(self.X[0])
#        dp12 = sin(self.X[0])
#        dp21 = 1
#        dp22 = self.X[1]
#        A1[0,0] = mean(dp11*dp11)
#        A1[1,1] = mean(dp12*dp12)
#        A1[0,1] = mean(dp11*dp12)
#        A1[1,0] = A1[0,1]
#        A2[0,0] = mean(dp21*dp21)
#        A2[1,1] = mean(dp22*dp22)  
#        A2[0,1] = mean(dp21*dp22)
#        A2[1,0] = A2[0,1]
#        return A1, A2
        
'============================================================================='        

#example functions
#def g1(N):
#    'g1 := equally distributed in the [0.7,1.3]'
#    return linspace(0.7,1.3,N)
#
#M = 6
#alpha = 2
#
#def h1(X, harg):
#    'h1 = c*cos(th) + s*sin(th) up to 0~2 th modes'
#    h = zeros([M,len(X)])
#    for j in range(M):
#        for m in range(3):
#            h[j] += harg[j,m,0]*cos(m*X)
#            h[j] += harg[j,m,1]*sin(m*X)
#    return h
#    
#def update_harg(X,y,h_hat,w):
#    'calculate dharg/dt'
#    harg = zeros([M,3,2])
#    for j in range(M):
#        for m in range(3):
#            harg[j,m,0] += mean(cos(m*X))*(y[j]-h_hat[j])*alpha*w
#            harg[j,m,1] += mean(sin(m*X))*(y[j]-h_hat[j])*alpha*w
#    return harg

    
    