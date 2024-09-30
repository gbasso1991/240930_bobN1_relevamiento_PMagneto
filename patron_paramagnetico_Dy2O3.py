#%%
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import pandas as pd
import chardet 
import re
from scipy.interpolate import interp1d
from uncertainties import ufloat, unumpy 
from scipy.optimize import curve_fit 
from scipy.stats import linregress
from uncertainties import ufloat, unumpy
from glob import glob

#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}
    
    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})  
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m
    
    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata

#%% 
directorio= os.getcwd()
ciclos_212 = glob(directorio,'212_150/*212kHz*.txt')
ciclos_238 = glob(os.path.join(directorio,'*238kHz*'))
ciclos_265 = glob(os.path.join(directorio,'*265kHz*'))
ciclos_300 = glob(os.path.join(directorio,'*300kHz*'))

# %%
t_135_1,H_Vs_135_1,M_Vs_135_1,H_kAm_135_1,M_Am_135_1,_= lector_ciclos(ciclos_135[0])
t_135_2,H_Vs_135_2,M_Vs_135_2,H_kAm_135_2,M_Am_135_2,_= lector_ciclos(ciclos_135[1])
t_135_3,H_Vs_135_3,M_Vs_135_3,H_kAm_135_3,M_Am_135_3,_= lector_ciclos(ciclos_135[2])

t_212_1,H_Vs_212_1,M_Vs_212_1,H_kAm_212_1,M_Am_212_1,_= lector_ciclos(ciclos_212[0])
t_212_2,H_Vs_212_2,M_Vs_212_2,H_kAm_212_2,M_Am_212_2,_= lector_ciclos(ciclos_212[1])
t_212_3,H_Vs_212_3,M_Vs_212_3,H_kAm_212_3,M_Am_212_3,_= lector_ciclos(ciclos_135[2])

t_300_1,H_Vs_300_1,M_Vs_300_1,H_kAm_300_1,M_Am_300_1,_= lector_ciclos(ciclos_300[0])
t_300_2,H_Vs_300_2,M_Vs_300_2,H_kAm_300_2,M_Am_300_2,_= lector_ciclos(ciclos_300[1])
t_300_3,H_Vs_300_3,M_Vs_300_3,H_kAm_300_3,M_Am_300_3,_= lector_ciclos(ciclos_300[2])
#%%
fig,ax = plt.subplots(figsize=(8,6),constrained_layout=True)
ax.plot(H_kAm_135_1,M_Vs_135_1)
ax.plot(H_kAm_135_2,M_Vs_135_2)
ax.plot(H_kAm_135_3,M_Vs_135_3)

ax.plot(H_kAm_212_1,M_Vs_212_1)
ax.plot(H_kAm_212_2,M_Vs_212_2)
ax.plot(H_kAm_212_3,M_Vs_212_3)

ax.plot(H_kAm_300_1,M_Vs_300_1)
ax.plot(H_kAm_300_2,M_Vs_300_2)
ax.plot(H_kAm_300_3,M_Vs_300_3)

ax.grid()
ax.set_ylabel('M (V*s)')
ax.set_xlabel('H (A/m)')
#%% Ajuste lineal sobre cada ciclo
def lineal(x,m,n):
    return m*x+n

(m_135_1,n_135_1),_ = curve_fit(f=lineal, xdata=H_kAm_135_1,ydata=M_Vs_135_1)
(m_135_2,n_135_2),_ = curve_fit(f=lineal, xdata=H_kAm_135_2,ydata=M_Vs_135_2)
(m_135_3,n_135_3),_ = curve_fit(f=lineal, xdata=H_kAm_135_3,ydata=M_Vs_135_3)

(m_212_1,n_212_1),_ = curve_fit(f=lineal, xdata=H_kAm_212_1,ydata=M_Vs_212_1)
(m_212_2,n_212_2),_ = curve_fit(f=lineal, xdata=H_kAm_212_2,ydata=M_Vs_212_2)
(m_212_3,n_212_3),_ = curve_fit(f=lineal, xdata=H_kAm_212_3,ydata=M_Vs_212_3)

(m_300_1,n_300_1),_ = curve_fit(f=lineal, xdata=H_kAm_300_1,ydata=M_Vs_300_1)
(m_300_2,n_300_2),_ = curve_fit(f=lineal, xdata=H_kAm_300_2,ydata=M_Vs_300_2)
(m_300_3,n_300_3),_ = curve_fit(f=lineal, xdata=H_kAm_300_3,ydata=M_Vs_300_3)

m_mean  = np.mean(np.array([m_135_1,m_135_2,m_135_3,m_212_1,m_212_2,m_212_3,m_300_1,m_300_2,m_300_3]))
m_std = np.std(np.array([m_135_1,m_135_2,m_135_3,m_212_1,m_212_2,m_212_3,m_300_1,m_300_2,m_300_3]))
m =ufloat(m_mean,m_std)
print(f'Pendiente media = {m:.2e} Vs/A/m')
n_mean  = np.mean(np.array([n_135_1,n_135_2,n_135_3,n_212_1,n_212_2,n_212_3,n_300_1,n_300_2,n_300_3]))

#%%
x_new= np.linspace(-57712,57712,100)
y_new= lineal(x_new,m_mean,n_mean)

fig,ax = plt.subplots(constrained_layout=True)
ax.plot(H_kAm_135_1,M_Vs_135_1)
ax.plot(H_kAm_135_2,M_Vs_135_2)
ax.plot(H_kAm_135_3,M_Vs_135_3)

ax.plot(H_kAm_212_1,M_Vs_212_1)
ax.plot(H_kAm_212_2,M_Vs_212_2)
ax.plot(H_kAm_212_3,M_Vs_212_3)

ax.plot(H_kAm_300_1,M_Vs_300_1)
ax.plot(H_kAm_300_2,M_Vs_300_2)
ax.plot(H_kAm_300_3,M_Vs_300_3)

ax.plot(x_new,y_new,'o-',label=f'$< m > =${m} Vs/A/m')
ax.legend()

ax.grid()
ax.set_ylabel('M (V*s)')
ax.set_xlabel('H (A/m)')
ax.set_title('Patron Dy$_2$O$_3$')
plt.show()