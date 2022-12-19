# -*- coding: utf-8 -*-
import math
import numpy as np 
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
import tools

class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return math.floor(x / self.discrete + 0.5)

class SourceBase(metaclass=ABCMeta):
    @abstractmethod
    def getE(self, m, q):
        pass

class GaussianDiff(SourceBase):
    """
    Класс с уравнением плоской волны
    для дифференцированного гауссова сигнала в дискретном виде
    d - определяет задержку сигнала.
    w - определяет ширину сигнала.
    Sc - число Куранта.
    eps - относительная диэлектрическая проницаемость среды,
    в которой расположен источник.
    mu - относительная магнитная проницаемость среды,
    в которой расположен источник.
    """

    def __init__(self, d, w, Sc=1.0, eps=1.0, mu=1.0):
        self.d = d
        self.w = w
        self.Sc = Sc
        self.eps = eps
        self.mu = mu

    def getE(self, m, q):
        """
        Расчет поля E в дискретной точке пространства m
        в дискретный момент времени q
        """
        tmp = ((q - m * np.sqrt(self.eps * self.mu) / self.Sc) - self.d) / self.w
        return -2 * tmp * np.exp(-(tmp ** 2))

if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * np.pi

    # Число Куранта
    Sc = 1.0

    # Скорость света
    c = 3e8

    # Время расчета в секундах
    maxTime_s = 50e-9

    # Размер области моделирования в метрах
    maxSize_m = 3.5

    #Размер ячейки разбиения
    dx = 5e-3

    # Размер области моделирования в отсчетах
    maxSize = int(maxSize_m / dx)

    #Шаг дискретизации по времени
    dt = Sc * dx / c

    sampler_x = Sampler(dx)
    sampler_t = Sampler(dt)

    #Время расчета в отсчетах
    maxTime = sampler_t.sample(maxTime_s)

    #Размер области моделирования в отсчетах
    maxSize = sampler_x.sample(maxSize_m)

    # Положение источника в отсчетах
    sourcePos = int(maxSize / 2)

    # Датчики для регистрации поля
    probesPos = [sourcePos + 50]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = np.ones(maxSize)
    eps[:] = 8.0

    # Магнитная проницаемость
    mu = np.ones(maxSize - 1)
    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)

    source = GaussianDiff(30.0, 14.0, Sc, eps[sourcePos], mu[sourcePos])

    # Ez[1] в предыдущий момент времени
    oldEzLeft = Ez[1]

    # Ez[-2] в предыдущий момент времени
    oldEzRight = Ez[-2]

    # Коэффициенты для расчета ABC второй степени
    # Sc' для левой границы
    tempLeft = Sc / np.sqrt(mu[0] * eps[0])
    koeffABCLeft = (tempLeft - 1) / (tempLeft + 1)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel, dx, dt)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for q in range(maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getE(0, q)

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (np.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getE(-0.5, q + 0.5))

         # Граничные условия ABC первой степени
        Ez[0] = oldEzLeft + koeffABCLeft * (Ez[1] - Ez[0])
        oldEzLeft = Ez[1]
       

        # Граничные условия PMC (справа)
        Ez[-1] = Ez[-2]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 50 == 0:
            display.updateData(display_field, q)

    display.stop()
    
    time_list = np.arange(0, len(probes[0].E)) * dt *1e9
    tlist = np.arange(0, maxTime * dt, dt)
    # Отображение сигнала, сохраненного в датчиках
    #tools.showProbeSignals(probes, -1.1, 1.1, dt)
    fig, ax = plt.subplots()
    ax.set_xlim(0, 50e-9)
    ax.set_ylim(-1.3, 1.1)
    ax.set_xlabel('t, нс')
    ax.set_ylabel('Ez, В/м')
    ax.plot(tlist, probe.E/np.max(probe.E))
    ax.minorticks_on()
    ax.grid()
    # Отображение спектра сигнала
    tools.FFT(probe.E, dt).spectr()
