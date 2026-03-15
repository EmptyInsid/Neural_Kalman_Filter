import torch

from app.neural_calman_app.neural.neural import (
    KalmanGainNet, NoiseEstimator,
    train_neural_kalman,
    train_noise_estimator
)

from app.neural_calman_app.model.model import (
    simulate_physical_motion,
    simulate_changing_noise_motion,
)


# model = KalmanGainNet()

# train_neural_kalman(
#     model,
#     simulate_physical_motion,
#     epochs=500
# )

# torch.save(
#     model.state_dict(),
#     "kalman_net.pth"
# )

# print("Model KalmanGainNet saved")

model = NoiseEstimator()

train_noise_estimator(
    model,
    simulate_changing_noise_motion,
    epochs=500
)

torch.save(
    model.state_dict(),
    "kalman_noise_net.pth"
)

print("Model NoiseEstimator saved")
