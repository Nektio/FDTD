""" 
Расчет зависимости эффективной площади рассеяния (ЭПР)идеально проводящей
сферы от частоты.
Построить график зависимости ЭПР от частоты.
Результат сохранить в текстовый файл формата JSON
"""
# Импортирование библиотек
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
from scipy.constants import pi, speed_of_light
from scipy.special import spherical_jn as jn
from scipy.special import spherical_yn as yn

#Формулы и обозначения по заданию
"""
sigma = lambda**2/pi*(|Σ(((-1)**n)*(n+0.5)*(bn - an))|**2)

an = jn(kr)/hn(kr)

bn = (kr*jn-1(kr)-n*jn(kr))/(kr*hn-1(kr)-n*hn(kr))

hn(x) = jn(x) + iyn(x)

• k — волновое число.
• l — длина волны.
• r — радиус сферы.
• jn(x) и yn(x) — сферические функции Бесселя первого и второго рода
соответственно порядка n.
• i — мнимая единица.
• hn(x) — сферическая функция Бесселя третьего рода.
"""

# Функции для упрощения записи
def hn(n, x):
    return jn(n, x) + 1j * yn(n, x)


def bn(n, x):
    return (x * jn(n - 1, x) - n * jn(n, x)) / (x * hn(n - 1, x) - n * hn(n, x))


def an(n, x):
    return jn(n, x) / hn(n, x)


url = "https://jenyay.net/uploads/Student/Modelling/task_02.xml"

#def get_html(url):
    #r = requests.get(url)
    #return r.text
result = requests.get(url)
if result.status_code == requests.codes.OK:
    file = result.text#
else:
    exit(1)

soup = BeautifulSoup(file, 'xml')
var = soup.find(attrs={"number": "7"})

# Присвоение значений из файла под нужным вариантом
D = float(var["D"])
fmin = float(var["fmin"])
fmax = float(var["fmax"])
step = 1e7

# Расчет значений
r = D / 2
freq = np.arange(fmin, fmax, step)
lambda_ = speed_of_light / freq
k = 2 * pi / lambda_

# Расчет суммы
arr_sum = [
    ((-1) ** n) * (n + 0.5) * (bn(n, k * r) - an(n, k * r)) for n in range(1, 50)
]
sum_ = np.sum(arr_sum, axis=0)
sigma = (lambda_ ** 2) / pi * (np.abs(sum_) ** 2)

dir_ = pathlib.Path("results")
dir_.mkdir(exist_ok=True)
file = dir_ / "results.json"


#with file.open("w") as f:
    #json.dump(result, f, indent=4)

with file.open("w") as f:
    f.write( json.dumps( {'freq':freq.tolist() , }))
    f.write("\n")
    f.write( json.dumps( {'lmbd':lambda_.tolist() }))
    f.write("\n")
    f.write( json.dumps( {'sigma':sigma.tolist() }))
plt.plot(freq / 10e6, sigma)
plt.xlabel("$f, МГц$")
plt.ylabel(r"$\sigma, м^2$")
plt.grid()
plt.savefig("results/results.png")
plt.show()