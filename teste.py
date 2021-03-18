import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import sqrt


def func(x, a, b):
    return a*(x**b)


def media(valores):
    return (sum(valores)/len(valores))


def desvio_padrao(valores):
    m = media(valores)
    nova_lista = []
    for v in valores:
        nova_lista.append((v-m)**2)

    valor = sqrt(sum(nova_lista))
    return valor/len(nova_lista)


massa = 5

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

x = np.array([media(values1), media(values2), media(values3), media(values4),
              media(values5), media(values6), media(values7), media(values8),
              media(values9), media(values10)])

y = np.array([massa, massa/2, massa/(2**2), massa/(2**3), massa/(2**4),
              massa/(2**5), massa/(2**6), massa/(2**7), massa/(2**8),
              massa/(2**9)])

plt.title('Fractais bolinha de papel')

yerr = 0.5
xerr = 0.05

plt.ylabel('Massa (g)')
plt.xlabel('Diâmetro médio (cm)')

plt.errorbar(x, y, xerr=xerr,  yerr=yerr, fmt=' ',
             color='black', label='Incertezas')

params, extras = curve_fit(func, x, y)

a, b = params

plt.plot(x, func(x, a, b), 'r',
         label=f'Parâmetros de treinamento: a={round(a, 3)}, b={round(b, 3)}' +
         '\nFunction: a*(x**b)')

plt.plot(x, y, marker='.', ls='', color='black', label='Dados experimentais')

plt.legend()
plt.show()
