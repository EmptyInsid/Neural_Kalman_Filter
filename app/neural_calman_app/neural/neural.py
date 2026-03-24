import numpy as np
import torch
import torch.nn as nn


# Neural network for Kalman Gain
class KalmanGainNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(4, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()

        )

    def forward(self, x):

        return self.net(x)
    
class NoiseEstimator(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),  # Only Tanh, GELU or Mish!
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, x):

        return self.net(x)


# Neural Kalman Filter implementation
def neural_kalman_filter(
        x,
        z,
        N,
        a,
        T,
        model,
        motion_type
):

    xOpt = np.zeros(N)
    K = np.zeros(N)

    vOpt = np.zeros(N)

    residual_history = [0, 0, 0, 0, 0]

    P = 1
    sigmaPsi = 1

    xOpt[0] = z[0]

    for t in range(N-1):

        # prediction
        if motion_type == "simple":

            x_pred = xOpt[t] + a*T*t

        else:

            vOpt[t+1] = vOpt[t] + a*T

            x_pred = (
                xOpt[t]
                + vOpt[t]*T
                + 0.5*a*T**2
            )

        residual = z[t+1] - x_pred

        # обновляем историю residual
        residual_history.pop(0)
        residual_history.append(residual)

        r = residual_history

        # признаки для сети
        features = torch.tensor(

            [
                r[0] / 10,
                r[1] / 10,
                r[2] / 10,
                r[3] / 10,
                r[4] / 10,
                x_pred / 100
            ],

            dtype=torch.float32

        )

        sigmaEta = model(features).item()

        # ограничиваем шум
        sigmaEta = max(0.1, min(sigmaEta, 20))

        # Kalman equations
        P = P + sigmaPsi**2 # добавляем шум процесса (не уверены, P большое)

        K_t = P / (P + sigmaEta**2) # насколько доверять датчику или модели

        K[t+1] = K_t

        xOpt[t+1] = (x_pred*(1-K_t) + K_t*z[t+1]) # корректировка состояния

        P = (1-K_t)*P # теперь более уверены в оценке, P уменьшается

    err = xOpt[:-1] - x[:-1]

    DZ = err * 100 / (np.max(x) - np.min(x))

    SKO = np.std(DZ)

    return xOpt, DZ, SKO, K


# Training procedure
def train_neural_kalman(
        model,
        simulate_function,
        epochs=1000
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_fn = nn.MSELoss()

    for epoch in range(epochs):

        sigmaPsi = np.random.uniform(0.1,5)
        sigmaEta = np.random.uniform(0.1,5)

        x, z = simulate_function(
            100,
            0.1,
            1,
            sigmaPsi,
            sigmaEta
        )

        xOpt = torch.zeros(100)

        prev_residual = 0

        losses = []

        for t in range(1,100):

            x_pred = xOpt[t-1] + 0.1*(t-1) # прогнозируем положение объекта

            residual = z[t] - x_pred # ошибка измерения

            features = torch.tensor(
                [z[t], x_pred, residual, prev_residual],
                dtype=torch.float32
            ) # формируем вход сети - по этим данным понять какой K лучше

            K = model(features)

            x_est = (1-K)*x_pred + K*z[t] # обновление состояния

            target = torch.tensor(
                x[t],
                dtype=torch.float32
            ) # знаем истинное положение объекта, потому что данные синтетические

            loss = loss_fn(x_est, target) # сеть штрафуется
            losses.append(loss)

            xOpt[t] = x_est.detach() # сохраняем состояние

            prev_residual = residual # обновляем residual

        total_loss = torch.stack(losses).mean() # берём среднюю ошибку по всей траектории

        # стандартный градиентный спуск
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:

            print(
                f"epoch {epoch} loss {total_loss.item():.4f}"
            )


def train_noise_estimator(
        model,
        simulate_function,
        epochs=1000
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    loss_fn = nn.MSELoss()

    for epoch in range(epochs):

        N = 100

        x, z, sigma = simulate_function(
            N,
            0.1,
            1,
            "physic"
        )

        residual_history = [0, 0, 0, 0, 0]

        losses = []

        for t in range(1, N):

            x_pred = x[t-1] + 0.1*(t-1)

            residual = z[t] - x_pred

            residual_history.pop(0)
            residual_history.append(residual)

            r = residual_history

            features = torch.tensor(
                [
                    r[0] / 10,
                    r[1] / 10,
                    r[2] / 10,
                    r[3] / 10,
                    r[4] / 10,
                    x_pred / 100
                ],

                dtype=torch.float32

            )

            sigma_pred = model(features)

            sigma_true = torch.tensor(
                [sigma[t]],
                dtype=torch.float32
            )

            loss = loss_fn(
                sigma_pred,
                sigma_true
            )

            losses.append(loss)

        total_loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(
                f"epoch {epoch} loss {total_loss.item():.4f}"
            )
