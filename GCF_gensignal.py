# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:51:38 2016

@author: jkim684
"""
from numpy import linspace, cos, mean, sqrt, zeros, array, pi, abs
from numpy import sum as npsum
from numpy.random import uniform, normal, randint, choice
import FPF
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button
#import multiprocessing as mp
from time import time
from copy import copy
#from ctypes import c_double

'== Control Panel =='
T_fin = 50.0
T_tran = 40.0
T_stop = 30.0
Do_Plot = True#False#
fpool = [0.250, sqrt(2)/4,0.5,sqrt(2)/2, 1.0,sqrt(2), 2.0]#[1/sqrt(6),1/sqrt(3),1/sqrt(2),1,sqrt(2),sqrt(3),2.0]
learn = True#False#
sWadapt = False#True#
filt = FPF.FPF_phase

'== Global Params =='
alp = 0.3 if learn else 0.0
bet = 0.1
N_glob = 50
dt_glob = 0.005
sB_glob = 0.1*sqrt(dt_glob)
sW_signal = 0.1*sqrt(dt_glob)#0.007
sW_filt = 0.3*sqrt(dt_glob)
M_glob = 1
gam_glob = 0.1
c0_glob = [0.0]
sc = 1.0
Nfilt = len(fpool)

minA = 6
maxA = 12

def hz2rad(inp):
    return 2*pi*array(inp)

'== FPF g,h,learning functions =='
def g1(N, w):
    'uniform distribution [w(1-g),w(1+g)]'
    return linspace(w*(1-gam_glob),w*(1+gam_glob),N)
    
def h1(X, harg):
    'c*cos(th)'
    h = zeros([1,X.size])
    h[0] = harg*cos(X)
    return h
    
def adapt1(X,y,hh,harg,w):
    'calculate dharg/dt'
    dharg = mean(cos(X))*(y[0]-hh[0])*alp*w - bet*harg
    return dharg

'==========================START MAIN PGM =========================='
if __name__ == '__main__':
    '== Get input from the user =='
    ss = 'Frequency pool:  {:5.3f}Hz'.format(fpool[0])
    for j in range(1, Nfilt): ss += ', {:5.3f}Hz'.format(fpool[j])
    print(ss)
    
    buf = input('\nNumber of freqs for input signal? (1-{:d})   '.format(len(fpool)))
    Nfreq = -1
    try: Nfreq = int(buf)
    except: pass
    if Nfreq <= 0 or Nfreq > len(fpool):
        print('Wrong input')
        raise SystemExit
    
    buf = input('Maximum perturbation? (Hz)   ')
    sf = -1.0
    try: sf = float(buf)
    except: pass    
    if sf < 0:
        print('Wrong input')
        raise SystemExit
    
    '== Initialize Coeffs =='
    a = zeros(Nfreq)
    for j in range(Nfreq):
        a[j] = randint(minA,maxA+1)/10
        if bool(randint(0,2)): a[j] *= -1
        
    f_index = choice(len(fpool),Nfreq,False)
    f_index.sort()
    a_full = zeros(Nfilt)
    for j in range(Nfreq): a_full[f_index[j]] = a[j]
    
    f = array([fpool[f_index[j]] for j in range(Nfreq)])
    ss = '\nSelected freq : {:5.3f}Hz'.format(f[0])
    for j in range(1, Nfreq): ss += ', {:5.3f}Hz'.format(f[j])
    print(ss)
    
    if sf > 0:
        f += sf*uniform(0.8,1.2,Nfreq)*choice([-1,1],Nfreq,True) #sf*uniform(-1.0,1.0,Nfreq)
        ss = 'Perturbed freq: {:5.3f}Hz'.format(f[0])
        for j in range(1, Nfreq): ss += ', {:5.3f}Hz'.format(f[j])
        print(ss)
    else:
        print('No perturbation')        
        
    ss = 'Generated amp :  {:4.1f}  '.format(a[0])
    for j in range(1, Nfreq): ss += ',  {:4.1f}  '.format(a[j])
    print(ss)
    
    w = hz2rad(f)
    w_full = hz2rad(fpool)
    
    buf = input('\nPress Enter to proceed')
    print('\n ... Running ... \n')
    stime = time()
    
    '== Make Signal =='
    NLine = int(T_fin/dt_glob)
    Nstop = int(T_stop/dt_glob)
    dZ = sW_signal*sqrt(dt_glob)*normal(0,1,NLine)
    for j in range(Nfreq):
        dX = w[j]*dt_glob + sB_glob*sqrt(dt_glob)*normal(0,1,NLine)
        X = array([npsum(dX[:i]) for i in range(NLine)])
        dZ += a[j]*cos(X)*dt_glob
        
    if Nstop > 0: dZ[Nstop:] = sW_signal*sqrt(dt_glob)*normal(0,1,NLine-Nstop)

    '== Run filter objects =='
    #shared.list()
    FPFs = [filt(uniform(-pi,pi,N_glob),M_glob,lambda N: g1(N,w_full[j]),
                      h1,sW_filt*sqrt(1/(npsum(1/w_full)*w_full[j])),sB_glob,
                      dt_glob,c0_glob if learn else [a_full[j]],adapt1,sc,sWadapt) for j in range(Nfilt)]
    yss = []
    hss = []
    rss = []
    css = []
    for j in range(Nfilt):
        yss.append([])
        hss.append([])
        css.append([])
        rss.append([])
                
    for i in range(NLine):
        h_c = [FPFs[j].h_hat for j in range(Nfilt)]
        for j in range(Nfilt):
            FPFs[j].run_step_y(array([dZ[i]/dt_glob-(npsum(h_c,axis=0)-h_c[j])]))
            yss[j].append(FPFs[j].y[0])
            hss[j].append(FPFs[j].h_hat[0])
            rss[j].append(FPFs[j].get_R())
            css[j].append(copy(FPFs[j].harg))
    hs = [npsum([hss[j][i] for j in range(Nfilt)]) for i in range(NLine)]
    ys = dZ/dt_glob
    N_tran = int(T_tran/dt_glob)
    l = [sqrt(mean(array(hss[j][N_tran:])*array(hss[j][N_tran:]))*2) for j in range(Nfilt)]
    c = [mean(css[j][N_tran:]) for j in range(Nfilt)]
    
    '== Visualization =='
    t = linspace(0,dt_glob*NLine,NLine)
    '-- signals --'
    fig = plt.figure(num=None, figsize=(5*(Nfilt+1),12),tight_layout=True)
    for j in range(Nfilt):
        ax = fig.add_subplot(3,Nfilt+1,j+1)
        ax.plot(t,yss[j],'r-',t,hss[j],'b-')
        ax.axhline(a_full[j],c='g',ls='--')
        ax.axhline(-a_full[j],c='g',ls='--')
        ax.grid()
        plt.ylim([-3,3])
        plt.title('Estimation of oscillator, {:5.3f}Hz'.format(fpool[j]))
    ax2 = fig.add_subplot(3,Nfilt+1,Nfilt+1)
    ax2.plot(t,ys,'r-',t,hs,'b-')
    ax2.grid()
    plt.ylim([-3,3])
    plt.title('Estimation of whole system')
    
    '-- order params --'
    for j in range(Nfilt):
        ax = fig.add_subplot(3,Nfilt+1,Nfilt+j+2)
        ax.plot(t,rss[j],'b-')
        ax.grid()
        plt.ylim([0.0,1.0])
        plt.title('Order parameter, {:5.3f}Hz'.format(fpool[j]))
    plt.show()
        
    '-- coefficients --'
    for j in range(Nfilt):
        ax = fig.add_subplot(3,Nfilt+1,2*Nfilt+j+3)
        ax.plot(t,css[j],'b-')
        ax.axhline(a_full[j],c='g',ls='--')
        ax.axhline(-a_full[j],c='r',ls='--')
        ax.grid()
        plt.ylim([-1.52,1.52])
        plt.title('Coefficients, {:5.3f}Hz'.format(fpool[j]))
    plt.show()
    
    '== Print out results =='
    ss = '{:5.3f}Hz'.format(fpool[f_index[0]])
    for j in range(1, Nfreq): ss += ', {:5.3f}Hz'.format(fpool[f_index[j]])
    ss += '\n{:5.3f}Hz'.format(f[0])
    for j in range(1, Nfreq): ss += ', {:5.3f}Hz'.format(f[j])
    ss += '\n {:4.1f}  '.format(a[0])
    for j in range(1, Nfreq): ss += ',  {:4.1f}  '.format(a[j])
    ss += '\nPerturbed: No' if sf == 0 else '\nPerturbed: Yes, {}'.format(sf)
    fig.text((Nfilt+0.05)/(Nfilt+1),0.61,ss,verticalalignment='top',size=13)
    
    ss = 'frequency  input  avg.mag  avg.c'
    for j in range(Nfilt):
        ss += '\n {:5.3f} Hz   {:4.1f}    {:7.4f}   {:6.3f}'.format(fpool[j],a_full[j],l[j],c[j])
    fig.text((Nfilt+0.05)/(Nfilt+1),0.53,ss,verticalalignment='top',size=13)
    
    print("Calculation took %f seconds" % (time() - stime))

    def set_lim(xmin,xmax):
        for ax in fig.axes:
            ax.set_xlim([xmin,xmax])
    axss = fig.add_axes([(Nfilt+0.1)/(Nfilt+1),0.64, 0.4/(Nfilt+1), 0.02])
    axss.set_xlim([0,T_fin])
    axss.set_yticks([0,1])
    sp = SpanSelector(axss, set_lim,'horizontal')
    
    def rst_lim(dummy):
        for ax in fig.axes:
            ax.set_xlim([0,T_fin])
    rstbtn = Button(plt.axes([(Nfilt+0.6)/(Nfilt+1),0.64, 0.35/(Nfilt+1), 0.02]),'Reset')
    rstbtn.on_clicked(rst_lim)