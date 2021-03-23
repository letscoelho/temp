import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b):
    return a*(x**b)

massa_zero = 5
errmassa_zero = 0.05 # de onde você tirou essa incerteza na massa?!?

values1 = np.array([4.6, 4, 4.4, 4, 4.7, 3.9, 4.3, 4.3])
values2 = np.array([2.6, 2.5, 2.7, 3.1, 2.5, 2.8, 2.9, 3])
values3 = np.array([1.9, 2.4, 2.3, 1.9, 2.3, 1.8, 2, 1.7])
values4 = np.array([1.6, 1.4, 1.5, 1.3, 1.4, 1.5, 1.5, 1.3])
values5 = np.array([1.3, 1.1, 1, 0.8, 1.2, 0.9, 0.9, 1])
values6 = np.array([0.6, 0.7, 0.7, 0.6, 0.7, 0.8, 0.6, 0.6])
values7 = np.array([0.5, 0.6, 0.7, 0.7, 0.7, 0.6, 0.7, 0.6])
values8 = np.array([0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4])
values9 = np.array([0.2, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2])
values10 = np.array([0, 0, 0, 0.1, 0.2, 0.1, 0, 0.1])

dadosdiam = [values1,values2,values3,values4,values5,values6,values7,values8,values9,values10]

diam_medio = []
massa = []
errdiam = []
errmassa = []

for i in range(1,11):
    diam_medio.append(np.average(dadosdiam[i-1]))
    massa.append(massa_zero/(2**(i-1)))
    errdiam.append(np.std(dadosdiam[i-1])/np.sqrt(len(dadosdiam[i-1])))
    errmassa.append(errmassa_zero/(2**(i-1)))
   
pinit = [2.5,2.5]
popt, pcov = curve_fit(func, diam_medio, massa, pinit)

   
plt.title = ('Fractais bolinha de papel')
plt.ylabel('Massa (g)')
plt.xlabel('Diâmetro médio (cm)')

plt.plot(diam_medio, massa, 'o', ls='', color='black', label='Dados experimentais')
plt.errorbar(diam_medio, massa, xerr=errdiam,  yerr=errmassa, fmt=' ', color='black')
plt.plot(np.arange(0.0, 5.0, 0.01), func(np.arange(0.0, 5.0, 0.01), *popt), 'r',
         label= "Parâmetros de ajuste função a*(x**b): /n a = %.4f"%popt[0] + "+/- %.4f"%pcov[0,0]+
         "/n a = %.4f"%popt[1] + "+/- %.4f"%pcov[1,1])
         
plt.legend()
plt.show()
