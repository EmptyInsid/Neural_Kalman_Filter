from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

CONFIG = {
    # time
    "dt": 0.1,
    "duration": 20,

    # signal
    "signal_type": "sin",

    "frame_size": 1024,
    "hop": 512,
    "bin_id": 20,

    # noise
    "noise_std": 0.5,

    # Kalman parameters
    "Q": 0.01,      # process noise
    "R": None,      # auto = noise_std^2
    "F": 1.0,
    "H": 1.0,

    # initial state
    "init_covariance": 1.0,

    "audio_input": "experiment/audio_base/input.wav",
    "audio_output": "experiment/audio_base/output_kalman.wav",
    "audio_comparison": "experiment/audio_base/bin_comparison.png"
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

    # ---------- Prediction ----------
    def predict(self):
        self.x0 = self.F * self.state
        self.p0 = self.F * self.covariance * self.F + self.Q

    # ---------- Correction ----------
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


# Audio download
sr, audio = wavfile.read(CONFIG["audio_input"])

audio = audio.astype(float)

# stereo translation -> mono
if audio.ndim > 1:
    audio = audio.mean(axis=1)

# add noisy
noise_std = 0.2
audio_noisy = audio + np.random.normal(
    0,
    noise_std*np.max(np.abs(audio)),
    len(audio)
)

audio = audio_noisy

# Splitting the signal into time windows (STFT)
frame_size = CONFIG["frame_size"]
hop = CONFIG["hop"]

frames = []

for i in range(0, len(audio)-frame_size, hop):
    frames.append(audio[i:i+frame_size])

frames = np.array(frames)


# Transition to the time-frequency domain
spectrogram = np.fft.rfft(frames, axis=1)


# Selecting a single frequency bin
bin_id = CONFIG["bin_id"]

# the amplitude of the selected frequency in time
measurements = np.abs(spectrogram[:, bin_id])


# Kalman amplitude filtering
R = CONFIG["R"] or CONFIG["noise_std"] **2

# start Calman
kf = KalmanFilter1D(
    q=CONFIG["Q"],
    r=R,
    f=CONFIG["F"],
    h=CONFIG["H"]
)

kf.set_state(measurements[0], 1)

filtered_amp = [
    kf.step(z) for z in measurements
]


# Phase Return
phase = np.angle(spectrogram[:, bin_id])

spectrogram[:, bin_id] = (
    np.array(filtered_amp)
    * np.exp(1j * phase)
)


# Reverse FFT
reconstructed = np.fft.irfft(spectrogram, axis=1)


# Overlap-Add
output = np.zeros(len(audio))

for i, frame in enumerate(reconstructed):
    start = i * hop
    output[start:start+frame_size] += frame


# Saving the result
wavfile.write(
    CONFIG["audio_output"],
    sr,
    output.astype(np.int16)
)


# Comparison of the filter operation
param_text = (
   f"noise_std={noise_std}\n"
   f"Q={CONFIG['Q']}\n"
   f"R={R:.4f}\n"
   f"F={CONFIG['F']}, H={CONFIG['H']}"
)

plt.figure(figsize=(10,5))
plt.plot(measurements, label="Noisy amplitude")
plt.plot(filtered_amp, linewidth=2,
         label="Kalman filtered")

plt.title(f"Kalman filtering of frequency bin {bin_id}")
plt.xlabel("Frame index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.gcf().text(
   0.08, 0.78,
   param_text,
   fontsize=10,
   bbox=dict(facecolor='white', alpha=0.8)
)
plt.tight_layout()
plt.savefig(CONFIG["audio_comparison"])

before_var = np.var(measurements)
after_var = np.var(filtered_amp)

print("Variance before:", before_var)
print("Variance after:", after_var)
