import matplotlib.pyplot as plt
import numpy as np
import pathlib

"""
# A = 512
# x = [-512;512]
  f(x)=(-(A+47)*(sin(sqrt(|(x / 2 + (A+47))|))))-x*(sin(sqrt(|(x - (A+47))|)))
"""
#Переменные
A = 512
x_min = -512
x_max = 512
dx = 0.5

#Функция
def i(x):
    return np.sin(np.sqrt(abs(x - (A+47))))

def g(x):
    return np.sin(np.sqrt(abs(x / 2 + (A+47))))

def f(x):
    return (-(A+47)*(g(x))-x*(i(x)))


x = np.arange(x_min, x_max+dx, dx)
y = f(x)


#Создаёт директорию по выбранному пути
res = pathlib.Path("results")
res.mkdir(exist_ok=True)
file = res / "results.txt"

with file.open("w") as f:
    for a, b in zip(x, y):
        f.write(f"{a}    {b}\n")


#Построение графика
plt.plot(x, y)
#Задание сетки
plt.grid()
#Сохранение графика в файле
plt.savefig("results.png")
#Отображение графика
plt.show()
