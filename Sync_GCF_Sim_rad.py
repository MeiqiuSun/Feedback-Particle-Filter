# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:31:08 2016
Serial FPF
@author: Jinwon Kim
"""
from numpy import linspace, cos, sqrt, zeros, array, cumsum, ones, dot, copy, exp#, sin, mean, sum
from numpy.random import uniform, normal
import FPF
import threading
import matplotlib.pyplot as plt
#from PyQt4 import QtGui
#import sys
from time import time, sleep
pi=3.14159265358979
radtohz = 1 #1 for rad/s, 2*pi for Hz

T = 60

scale = 1.0
SNR = 20
'>45dB : almost no noise / 20dB : some noise level / 0dB : noise power is the same of signal / negative SNR: tough noisy signal'

w1 = 1.2*radtohz
w2 = sqrt(7)*radtohz
w3 = sqrt(15)*radtohz
w4 = 4.0*radtohz

A1 = -0.6
A2 = 0.0
A3 = 0.7
A4 = 0.0

alp = 0.3
bet = 0.00

N_glob = 200
dt_glob = 0.02
sB_glob = 0.2*sqrt(dt_glob)
sW = sqrt((A1**2+A2**2+A3**2+A4**2)/(2*dt_glob*10**(SNR/10)))
sW_filt = max(min(sW,1),0.1)
'SNR := 10log(Var(sig)/Var(noise)) = 10log(sum[(A^2)/2]/sW^2)'
'To avoid too large/small gain, sigma_W put into the filter will be limited and scaled *I just arbitrarily set this'

w_min = 0.5*radtohz
w_max = 5.0*radtohz
M_glob = 1


def g2(N):
    return linspace(w_min,w_max,N)   #Dw = 0.05 => l = log(2)/2Dw = 6.9314718
        
def h4(X,harg): #learn first mode, c*cos(X), M = 1
    return harg*cos(X).reshape([1,X.size])
    
def ad1(X,y,hh,harg,E,sum_E):
     return dot(E,cos(X))/sum_E * (y[0]-hh[0])*alp - harg*bet  #*w
    
'==========================START MAIN PGM =========================='
#app = QtGui.QApplication(sys.argv)
if __name__ == '__main__':
    '==Generate signal=='
    LineNum = int(T/dt_glob)
    t = array([i*dt_glob for i in range(LineNum)])
    dX1 = w1*dt_glob + sB_glob*sqrt(dt_glob)*normal(0,1,LineNum)
    dX2 = w2*dt_glob + sB_glob*sqrt(dt_glob)*normal(0,1,LineNum)
    dX3 = w3*dt_glob + sB_glob*sqrt(dt_glob)*normal(0,1,LineNum)
    dX4 = w4*dt_glob + sB_glob*sqrt(dt_glob)*normal(0,1,LineNum)
    
    #dX1[:1500] = 1.0*dt_glob + sB_glob*sqrt(dt_glob)*normal(0,1,1500)
    #dX1[1000:2000] = sB_glob*sqrt(dt_glob)*normal(0,1,1000)
    #dX1 = (w1 + 0.5*sin(0.2*t))*dt_glob + sB_glob*sqrt(dt_glob)*normal(0,1,LineNum)
    
    X1 = cumsum(dX1)
    X2 = cumsum(dX2)
    X3 = cumsum(dX3)
    X4 = cumsum(dX4)
        
    #dZ1 = array([A1*(0 if cos(x1)==0 else cos(x1)/abs(cos(x1)))*dt_glob for x1 in X1])
    #dZ2 = array([A2*(0 if cos(x2)==0 else cos(x2)/abs(cos(x2)))*dt_glob for x2 in X2])
    y1 = A1*cos(X1)
    y2 = A2*cos(X2)
    y3 = A3*cos(X3)
    y4 = A4*cos(X4)
    
    #y1[2000:4000] = zeros(2000)
    #dZ2[3000:] = zeros(3000)
    #dZ3[:3000] = zeros(3000)
    #dZ4[:2000] = zeros(2000)
    
    y_n = y1 + y2 + y3 + y4
    y = y_n + sW*sqrt(dt_glob)*normal(0,1,LineNum)
    #dZ[2000:4000] = sW*normal(0,1,2000)    
    
    '==Build FPF=='
    X00 = uniform(-pi,pi,N_glob)
    FPF1 = FPF.SC_GCF(X00,M_glob,g2,h4,sW_filt,sB_glob,dt_glob,0.2*ones(N_glob),ad1,scale)
    #QtApp = FPF.Qt_FPF_Plot(FPF1)
    thres = 0.109*N_glob
    thres2 = exp(-FPF1.l*((w_max-w_min)/(2*N_glob))**2)
    
    '==Run=='
    hs1 = []
    rots = []
    hargs = []
    hhs = []
    dt_trd = 0#dt_glob
    def trd_work():
        time0 = time()
        for i in range(LineNum):
            hs1.append(FPF1.h_hat[0])
            hhs.append(FPF1.hh_wtd[0])
            FPF1.run_step_y(array([y[i]]))
            #QtApp.append_data(dt_glob*i)
            E = FPF1.calc_E(FPF1.rho)
            a = (E.sum(axis=1)>thres)*((E>thres2).sum(axis = 1) > 2)*dot(E,FPF1.harg)/E.sum(axis=1)
            hargs.append(a)
            rots.append(copy(FPF1.rho))
            time1 = time()
            slp = dt_trd - time1 + time0
            if slp > 0: sleep(slp)
            time0 = time1
    #End if
    
    trd = threading.Thread(target=trd_work)
    trd.daemon = True
    trd.start()    
    #app.exec_()
    #QtApp.killTimer(1)
    dt_trd = 0.0
    trd.join()

    '==Plot Result=='
    fig1 = plt.figure(num=None, figsize=(12,11),tight_layout=True)
    ax11 = fig1.add_subplot(211)
    ax11.plot(t,y,'r-',t,hs1,'b-')
    ax11.grid()
    ax11.set_ylim([-3,3])
    ax11.set_xlabel('t')
    ax11.set_ylabel('y')
    ax11.set_title('Signal input and Estimation over time, SNR={}dB'.format(SNR))
    
    ax12 = fig1.add_subplot(212)
    ax12.plot(t,rots,'-',color='b',alpha=0.3)
    ax12.set_ylim([w_min,w_max])
    if A1 != 0.0: ax12.axhline(w1,ls='--',c='r',lw=3,alpha=0.7)
    if A2 != 0.0: ax12.axhline(w2,ls='--',c='r',lw=3,alpha=0.7)
    if A3 != 0.0: ax12.axhline(w3,ls='--',c='r',lw=3,alpha=0.7)
    if A4 != 0.0: ax12.axhline(w4,ls='--',c='r',lw=3,alpha=0.7)
    ax12.set_xlabel('t')
    ax12.set_ylabel(r'$\rho$')
    ax12.set_title('Rotation number of particles over time')
    ax12.grid()
    
    rho = copy(FPF1.rho)
    rho.sort()
    E = FPF1.calc_E(rho)
    fig2 = plt.figure(num=None, figsize=(5,5),tight_layout=True)
    ax21 = fig2.add_subplot(111)
    ax21.pcolor(FPF1.ws,FPF1.ws,E,cmap='gray')
    ax21.set_ylim([w_max,w_min])
    ax21.set_xlim([w_min,w_max])
    #plt.gca().invert_yaxis()
    ax21.set_title('Weight Matrix at t={}'.format(T))
    
    hargs = array(hargs)
    fig3 = plt.figure(num=None, figsize=(12,5),tight_layout=True)
    ax31 = fig3.add_subplot(111)
    ax31.plot(t,hargs,'.',color='b',alpha=0.1)
    ax31.grid()
    
    fig4 = plt.figure(num=None, figsize=(12,5),tight_layout=True)
    ax4 = fig4.add_subplot(111)
    ax4.plot(t,hhs,'-',color='b',alpha=0.1)
    if A1 != 0.0: ax4.axhline(abs(A1),ls='-',c='r')
    if A2 != 0.0: ax4.axhline(abs(A2),ls='-',c='r')
    if A3 != 0.0: ax4.axhline(abs(A3),ls='-',c='r')
    if A4 != 0.0: ax4.axhline(abs(A4),ls='-',c='r')
    ax4.grid()
    ax4.set_xlim([T-10,T])
    ax4.set_ylim([-3,3])
    ax4.set_xlabel('t')
    ax4.set_ylabel('y')
    ax4.set_title('Estimation of clusters over time')
        
    plt.show()
    
    '==Print Result=='
    print('cluster freqs:')
    sE = E.sum(axis=1)
    chkE = ((E>thres2).sum(axis=1) > 2)*(sE > thres)
    avR = dot(E,rho)/sE
    printed = -1
    for i in range(sE.size-1):
        #if sE[i] > thres:
        if chkE[i] & ((avR[i] - printed) >(w_max-w_min)/(N_glob)):
            #if (avR[i] if i == 0 else avR[i]-avR[i-1]) > (w_max-w_min)/(2*N_glob):
            #if i - index > 1:
            print('{:5.03f} / cluster size: {:5.02f}'.format(avR[i]/radtohz,sE[i]))
            printed = avR[i]
            #index = i
                
    print('actual freqs:')
    if A1 != 0.0: print('{:5.03f}'.format(w1/radtohz))
    if A2 != 0.0: print('{:5.03f}'.format(w2/radtohz))        
    if A3 != 0.0: print('{:5.03f}'.format(w3/radtohz))        
    if A4 != 0.0: print('{:5.03f}'.format(w4/radtohz))