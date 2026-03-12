import numpy as np
import matplotlib.pyplot as plt

CONFIG = {
    # time
    "dt": 0.1,
    "duration": 20,

    # signal
    "signal_type": "sin",

    # noise
    "noise_std": 0.5,

    # Kalman parameters
    "Q": 0.01,      # process noise
    "R": None,      # auto = noise_std^2
    "F": 1.0,
    "H": 1.0,

    # initial state
    "init_covariance": 1.0,
}


class KalmanFilter1D:

    def __init__(self, q, r, f=1.0, h=1.0):
        self.Q = q
        self.R = r
        self.F = f
        self.H = h

        self.state = 0.0
        self.covariance = 1.0

    def set_state(self, state, covariance):
        self.state = state
        self.covariance = covariance

    def predict(self):
        self.x0 = self.F * self.state
        self.p0 = self.F * self.covariance * self.F + self.Q

    def update(self, measurement):
        K = self.H * self.p0 / (
            self.H * self.p0 * self.H + self.R
        )

        self.state = self.x0 + K * (
            measurement - self.H * self.x0
        )

        self.covariance = (1 - K * self.H) * self.p0

    def step(self, measurement):
        self.predict()
        self.update(measurement)
        return self.state


# setting with config
dt = CONFIG["dt"]
t = np.arange(0, CONFIG["duration"], dt)

true_signal = np.sin(t)

noise_std = CONFIG["noise_std"]
measurements = true_signal + \
    np.random.normal(0, noise_std, len(t))

R = CONFIG["R"] or noise_std**2


# start Calman
kf = KalmanFilter1D(
    q=CONFIG["Q"],
    r=R,
    f=CONFIG["F"],
    h=CONFIG["H"]
)

kf.set_state(
    measurements[0],
    CONFIG["init_covariance"]
)

filtered = [kf.step(z) for z in measurements]

# grapf
param_text = (
    f"dt={CONFIG['dt']}\n"
    f"noise_std={noise_std}\n"
    f"Q={CONFIG['Q']}\n"
    f"R={R:.4f}\n"
    f"F={CONFIG['F']}, H={CONFIG['H']}\n"
    f"P0={CONFIG['init_covariance']}"
)

plt.figure(figsize=(10,5))

plt.plot(t, true_signal, label="True signal")
plt.plot(t, measurements, alpha=0.4, label="Noisy")
plt.plot(t, filtered, linewidth=2, label="Kalman output")

plt.legend()

plt.title("1D Kalman Filter Signal Denoising")

plt.gcf().text(
    0.87, 0.12,
    param_text,
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8)
)

plt.tight_layout()
plt.savefig("experiment/classic/kalman_result.png", dpi=150)

print("Save: kalman_result.png")

