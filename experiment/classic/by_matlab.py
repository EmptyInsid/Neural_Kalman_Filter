import numpy as np
import matplotlib.pyplot as plt

N = 100     # кол-во измерений
a = 0.15    # ускорение
T = -0.5    # шаг времени

sigmaPsi = 1      # шум модели
sigmaEta = 4      # шум датчика

# массив истинных координат
k = np.arange(N) 
x = np.zeros(N)
z = np.zeros(N)

# первое измерение: истинная координата + шум
z[0] = x[0] + np.random.normal(0, sigmaEta)

# генерация движения
for t in range(N-1):

    # x(t) + движение + шум модели
    x[t+1] = (
        x[t]
        + a*T*t
        + np.random.normal(0, sigmaPsi)
    )

    # x(t+1) + шум датчика
    z[t+1] = (
        x[t+1]
        + np.random.normal(0, sigmaEta)
    )

# ===== Фильтр Калмана =====

xOpt = np.zeros(N)
eOpt = np.zeros(N)
K = np.zeros(N)

# начальное значение равно измерению
xOpt[0] = z[0]      # начальная оценка
eOpt[0] = sigmaEta  # начальная ошибка оценки

for t in range(N-1):

    # обновление дисперсии ошибки оценки
    eOpt[t+1] = np.sqrt(
        (sigmaEta**2 * (eOpt[t]**2 + sigmaPsi**2))
        /
        (sigmaEta**2 + eOpt[t]**2 + sigmaPsi**2)
    )

    # коэффициент Калмана
    K[t+1] = (eOpt[t+1]**2) / sigmaEta**2

    # обновление состояния
    x_pred = xOpt[t] + a*T*t

    xOpt[t+1] = (
        x_pred*(1-K[t+1])
        + K[t+1]*z[t+1]
    )

# ===== график =====

plt.figure()

plt.plot(k, xOpt, label="Kalman estimate")
plt.plot(k, z, label="Sensor measurement")
plt.plot(k, x, "--", label="True trajectory")

plt.xlabel("time")
plt.ylabel("position")
plt.title("Kalman filtering result")
plt.legend()

plt.tight_layout()
plt.savefig("experiment/classic/by_matlab_res.png", dpi=150)
print("Save: by_matlab_res.png")

# ===== ошибка =====

# ошибка между оценкой Калмана и измерением
err = xOpt[:-1] - z[:-1]

DZ = err * 100 / (np.max(z) - np.min(z))

SKO = np.std(DZ)

print("SKO =", SKO)

plt.figure()

plt.plot(DZ)
plt.title("Filtering error (%)")

plt.tight_layout()
plt.savefig("experiment/classic/by_matlab_sko.png", dpi=150)
print("Save: by_matlab_sko.png")
