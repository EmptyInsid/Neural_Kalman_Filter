import sys
import os
import torch
import torch.nn as nn
import numpy as np
import time
import traceback

from PyQt6.QtCore import QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup,
    QPushButton, QDoubleSpinBox, QSpinBox, QFormLayout, QGroupBox, QCheckBox, QLabel
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from app.neural_calman_app.model.model import (
    simulate_motion,
    simulate_physical_motion,
    kalman_filter,
    kalman_full_experiment,
    simulate_changing_noise_motion
)
from app.neural_calman_app.neural.neural import (
    NoiseEstimator,
    neural_kalman_filter
)
from app.neural_calman_app.neural.train_model import hvp



def get_flat_params(model):
    return torch.cat([p.contiguous().view(-1) for p in model.parameters()])


def set_flat_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view_as(p))
        offset += numel


def cg_solve(f_Hx, b, cg_iters=15, residual_tol=1e-8):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = torch.dot(r, r)

    for i in range(cg_iters):
        Ap = f_Hx(p)
        curvature = torch.dot(p, Ap)


        if curvature <= 1e-8:
            if i == 0:

                x = b.clone()
            break

        alpha = rsold / curvature
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)

        if torch.sqrt(rsnew) < residual_tol:
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x

class TrainingThread(QThread):
    progress_signal = pyqtSignal(int, float)
    finished_signal = pyqtSignal(str, list)
    error_signal = pyqtSignal(str)

    def __init__(self, optimizer_name, epochs):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.epochs = epochs

    def run(self):
        try:

            N_train = 1000
            a = 0.15
            T = -0.5

            x_true, z, sigma = simulate_changing_noise_motion(N_train, a, T, "physical")

            X_list = []
            y_list = []

            xOpt = np.zeros(N_train)
            vOpt = np.zeros(N_train)
            xOpt[0] = z[0]

            for t in range(N_train - 1):
                vOpt[t + 1] = vOpt[t] + a * T
                x_pred = xOpt[t] + vOpt[t] * T + 0.5 * a * T ** 2


                f1 = z[t + 1]
                f2 = z[t]
                f3 = x_pred
                f4 = xOpt[t]
                f5 = vOpt[t + 1]
                f6 = z[t + 1] - x_pred

                X_list.append([f1, f2, f3, f4, f5, f6])

                y_list.append([sigma[t + 1]])

                K_approx = 0.2
                xOpt[t + 1] = x_pred * (1 - K_approx) + K_approx * z[t + 1]

            X = torch.tensor(X_list, dtype=torch.float32)
            y = torch.tensor(y_list, dtype=torch.float32)

            X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)

            model = NoiseEstimator()
            loss_fn = nn.MSELoss()
            loss_history = []

            if self.optimizer_name == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                for epoch in range(self.epochs):
                    optimizer.zero_grad()
                    preds = model(X)
                    loss = loss_fn(preds, y)
                    loss.backward()
                    optimizer.step()

                    loss_history.append(loss.item())
                    if epoch % 10 == 0:
                        self.progress_signal.emit(epoch, loss.item())

                torch.save(model.state_dict(), "kalman_noise_net_adam.pth")


            elif self.optimizer_name == "ER":


                lr_er = 0.05
                damping = 1.0
                cg_max_iters = 15

                for epoch in range(self.epochs):

                    preds = model(X)
                    loss = loss_fn(preds, y)
                    loss_history.append(loss.item())
                    params = list(model.parameters())
                    grads = torch.autograd.grad(loss, params, create_graph=True)
                    flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])

                    grad_norm = torch.norm(flat_grads)

                    if grad_norm > 1.0:
                        flat_grads = flat_grads / grad_norm

                    def hvp_damped(v):
                        h_v = hvp(loss, params, v)
                        return h_v + damping * v

                    step_direction = cg_solve(hvp_damped, -flat_grads, cg_iters=cg_max_iters)

                    current_weights = get_flat_params(model)
                    new_weights = current_weights + lr_er * step_direction

                    set_flat_params(model, new_weights)

                    self.progress_signal.emit(epoch, loss.item())

                torch.save(model.state_dict(), "kalman_noise_net_er.pth")

            self.finished_signal.emit(self.optimizer_name, loss_history)

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(f"Критическая ошибка в потоке обучения:\n{err_msg}")
            self.error_signal.emit(str(e))

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(16, 8))
        super().__init__(self.fig)


class KalmanApp(QWidget):
    def __init__(self):
        super().__init__()

        self.resize(1800, 1200)
        self.setWindowTitle("Kalman Filter Demo")
        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #dddddd; font-size: 12pt; }
            QGroupBox { border: 1px solid #444; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
            QPushButton { background-color: #2d89ef; color: white; border-radius: 6px; padding: 8px; }
            QPushButton:hover { background-color: #4aa3ff; }
            QPushButton:pressed { background-color: #1b5fa7; }
            QPushButton:disabled { background-color: #555555; color: #888888; }
            QSpinBox, QDoubleSpinBox { background-color: #2a2a2a; border: 1px solid #444; padding: 3px; }
        """)

        self.model_adam = NoiseEstimator()
        self.model_er = NoiseEstimator()
        self.loss_histories = {"Adam": [], "ER": []}

        if os.path.exists("kalman_noise_net_adam.pth"):
            try:
                self.model_adam.load_state_dict(torch.load("kalman_noise_net_adam.pth"))
            except RuntimeError as e:
                print(f"Не удалось загрузить веса Adam (возможно, изменилась архитектура). Ожидается переобучение.")
        self.model_adam.eval()

        if os.path.exists("kalman_noise_net_er.pth"):
            try:
                self.model_er.load_state_dict(torch.load("kalman_noise_net_er.pth"))
            except RuntimeError as e:
                print(f"Не удалось загрузить веса ER (возможно, изменилась архитектура). Ожидается переобучение.")
        self.model_er.eval()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)

        main_layout = QHBoxLayout()
        control_panel = QVBoxLayout()

        form = QFormLayout()
        self.N = QSpinBox();
        self.N.setRange(10, 10000);
        self.N.setValue(100)
        self.a = QDoubleSpinBox();
        self.a.setRange(-10, 10);
        self.a.setDecimals(3);
        self.a.setValue(0.15)
        self.T = QDoubleSpinBox();
        self.T.setRange(-10, 10);
        self.T.setDecimals(3);
        self.T.setValue(-0.5)
        self.sigmaPsi = QDoubleSpinBox();
        self.sigmaPsi.setRange(0, 100);
        self.sigmaPsi.setValue(1)
        self.sigmaEta = QDoubleSpinBox();
        self.sigmaEta.setRange(0, 100);
        self.sigmaEta.setValue(4)

        form.addRow("Number of steps (N)", self.N)
        form.addRow("Acceleration (a)", self.a)
        form.addRow("Time step (T)", self.T)
        form.addRow("Model noise σψ", self.sigmaPsi)
        form.addRow("Sensor noise ση", self.sigmaEta)

        params_box = QGroupBox("Simulation parameters")
        params_box.setLayout(form)
        control_panel.addWidget(params_box)

        model_box = QGroupBox("Motion model")
        model_layout = QVBoxLayout()
        self.model_classic = QRadioButton("Simple model")
        self.model_physics = QRadioButton("Physical motion")
        self.model_classic.setChecked(True)
        model_layout.addWidget(self.model_classic)
        model_layout.addWidget(self.model_physics)
        model_box.setLayout(model_layout)
        control_panel.addWidget(model_box)

        filter_box = QGroupBox("Filter type")
        filter_layout = QVBoxLayout()
        self.filter_kalman = QRadioButton("Classical Kalman")
        self.filter_neural = QRadioButton("Neural Kalman")
        self.filter_kalman.setChecked(True)
        filter_layout.addWidget(self.filter_kalman)
        filter_layout.addWidget(self.filter_neural)
        filter_box.setLayout(filter_layout)
        control_panel.addWidget(filter_box)

        train_box = QGroupBox("Neural Network Training")
        train_layout = QVBoxLayout()

        self.radio_adam = QRadioButton("Adam Optimizer")
        self.radio_er = QRadioButton("ER Optimizer")
        self.radio_adam.setChecked(True)

        self.btn_train = QPushButton("Train Network")
        self.btn_train.clicked.connect(self.start_training)

        self.btn_compare_conv = QPushButton("Show Convergence History")
        self.btn_compare_conv.clicked.connect(self.show_convergence)

        self.lbl_train_status = QLabel("Ready")

        self.cb_compare_opts = QCheckBox("Compare Adam and ER trajectories")

        train_layout.addWidget(self.radio_adam)
        train_layout.addWidget(self.radio_er)
        train_layout.addWidget(self.btn_train)
        train_layout.addWidget(self.lbl_train_status)
        train_layout.addWidget(self.btn_compare_conv)
        train_layout.addWidget(self.cb_compare_opts)

        train_box.setLayout(train_layout)
        control_panel.addWidget(train_box)

        self.button = QPushButton("Run simulation")
        self.button.clicked.connect(self.run_simulation)
        control_panel.addWidget(self.button)

        self.exp_button = QPushButton("Run parameter experiment for classic")
        self.exp_button.clicked.connect(self.run_experiment)
        control_panel.addWidget(self.exp_button)

        self.noise_exp_button = QPushButton("Run changing noise experiment")
        self.noise_exp_button.clicked.connect(self.run_changing_noise_experiment)
        control_panel.addWidget(self.noise_exp_button)

        control_panel.addStretch()

        self.canvas = MplCanvas()
        main_layout.addLayout(control_panel, 1)
        main_layout.addWidget(self.canvas, 4)
        self.setLayout(main_layout)

    def start_training(self):
        opt_name = "Adam" if self.radio_adam.isChecked() else "ER"
        epochs = 500 if opt_name == "Adam" else 100

        self.btn_train.setEnabled(False)
        self.lbl_train_status.setText(f"Training {opt_name}...")

        self.thread = TrainingThread(opt_name, epochs)
        self.thread.progress_signal.connect(self.update_training_progress)
        self.thread.finished_signal.connect(self.training_finished)
        self.thread.error_signal.connect(self.training_error)  # ПОДКЛЮЧАЕМ СИГНАЛ
        self.thread.start()

    def update_training_progress(self, epoch, loss):
        self.lbl_train_status.setText(f"Epoch {epoch} - Loss: {loss:.6f}")

    def training_finished(self, opt_name, loss_history):
        self.loss_histories[opt_name] = loss_history
        self.lbl_train_status.setText(f"{opt_name} Training Complete!")
        self.btn_train.setEnabled(True)

        if opt_name == "Adam":
            self.model_adam.load_state_dict(torch.load("kalman_noise_net_adam.pth"))
            self.model_adam.eval()
        else:
            self.model_er.load_state_dict(torch.load("kalman_noise_net_er.pth"))
            self.model_er.eval()

    def show_convergence(self):
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)

        if self.loss_histories["Adam"]:
            ax.plot(self.loss_histories["Adam"], label="Adam Convergence", color="#00d4ff")
        if self.loss_histories["ER"]:
            ax.plot(self.loss_histories["ER"], label="ER Convergence", color="#ff9900")

        if not self.loss_histories["Adam"] and not self.loss_histories["ER"]:
            ax.text(0.5, 0.5, "No training history found.\nPlease train models first.",
                    ha='center', va='center', fontsize=14)
        else:
            ax.set_title("Optimizer Convergence Comparison")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss (MSE)")
            ax.set_yscale('log')
            ax.legend()

        self.canvas.draw()

    def run_simulation(self):
        N = self.N.value()
        a = self.a.value()
        T = self.T.value()
        sigmaPsi = self.sigmaPsi.value()
        sigmaEta = self.sigmaEta.value()

        motion_model = simulate_motion if self.model_classic.isChecked() else simulate_physical_motion
        self.x, self.z = motion_model(N, a, T, sigmaPsi, sigmaEta)
        motion_type = "simple" if self.model_classic.isChecked() else "physical"

        self.k = np.arange(N)
        self.frame = 0
        self.canvas.fig.clear()
        self.canvas.ax1 = self.canvas.fig.add_subplot(211)
        self.canvas.ax2 = self.canvas.fig.add_subplot(212)

        ax = self.canvas.ax1
        ax.set_xlim(0, N)
        ax.set_ylim(min(self.x.min(), self.z.min()) - 5, max(self.x.max(), self.z.max()) + 5)
        ax.set_ylabel("position")

        self.true_line, = ax.plot([], [], color="black", label="True trajectory")
        self.sensor_line, = ax.plot([], [], "--", color="#ea02ff", label="Sensor")
        self.true_point, = ax.plot([], [], "o", color="black", markersize=8)
        self.sensor_point, = ax.plot([], [], "x", color="#ea02ff", markersize=8)

        self.compare_mode = False

        if self.filter_kalman.isChecked():
            ax.set_title(f"Classical Kalman")
            self.xOpt, self.DZ, self.SKO, self.K = kalman_filter(self.x, self.z, N, a, T, sigmaPsi, sigmaEta,
                                                                 motion_type)
            self.kalman_line, = ax.plot([], [], color="#00d4ff", label="Classical Filtered")
            self.kalman_point, = ax.plot([], [], "o", color="#00d4ff", markersize=8)

            self.canvas.ax2.plot(self.DZ, color="#00d4ff")
            self.canvas.ax2.set_title(f"Error %, SKO={self.SKO:.2f}")

        else:
            if self.cb_compare_opts.isChecked():
                self.compare_mode = True
                ax.set_title("Neural Kalman: Adam vs ER")

                self.xOpt_adam, self.DZ_adam, self.SKO_adam, _ = neural_kalman_filter(self.x, self.z, N, a, T,
                                                                                      self.model_adam, motion_type)
                self.xOpt_er, self.DZ_er, self.SKO_er, _ = neural_kalman_filter(self.x, self.z, N, a, T, self.model_er,
                                                                                motion_type)

                self.kalman_line_adam, = ax.plot([], [], color="#00d4ff", label="Neural (Adam)")
                self.kalman_line_er, = ax.plot([], [], color="#ff9900", label="Neural (ER)")
                self.kalman_point_adam, = ax.plot([], [], "o", color="#00d4ff", markersize=6)
                self.kalman_point_er, = ax.plot([], [], "o", color="#ff9900", markersize=6)

                self.canvas.ax2.plot(self.DZ_adam, color="#00d4ff", label=f"Adam Error (SKO={self.SKO_adam:.2f})")
                self.canvas.ax2.plot(self.DZ_er, color="#ff9900", label=f"ER Error (SKO={self.SKO_er:.2f})")
                self.canvas.ax2.legend()
                self.canvas.ax2.set_title("Error Comparison")

            else:
                selected_model = self.model_adam if self.radio_adam.isChecked() else self.model_er
                opt_name = "Adam" if self.radio_adam.isChecked() else "ER"
                ax.set_title(f"Neural Kalman ({opt_name})")

                self.xOpt, self.DZ, self.SKO, self.K = neural_kalman_filter(self.x, self.z, N, a, T, selected_model,
                                                                            motion_type)
                self.kalman_line, = ax.plot([], [], color="#00d4ff", label=f"Neural Filtered ({opt_name})")
                self.kalman_point, = ax.plot([], [], "o", color="#00d4ff", markersize=8)

                self.canvas.ax2.plot(self.DZ, color="#00d4ff")
                self.canvas.ax2.set_title(f"Error %, SKO={self.SKO:.2f}")

        ax.legend()
        self.canvas.ax2.set_xlabel("time")
        self.canvas.ax2.set_ylabel("error %")
        self.canvas.draw()
        self.timer.start(16)

    def update_animation(self):
        if self.frame >= len(self.k):
            self.timer.stop()
            return

        self.true_line.set_data(self.k[:self.frame], self.x[:self.frame])
        self.sensor_line.set_data(self.k[:self.frame], self.z[:self.frame])
        self.true_point.set_data([self.k[self.frame]], [self.x[self.frame]])
        self.sensor_point.set_data([self.k[self.frame]], [self.z[self.frame]])

        if self.compare_mode:
            self.kalman_line_adam.set_data(self.k[:self.frame], self.xOpt_adam[:self.frame])
            self.kalman_line_er.set_data(self.k[:self.frame], self.xOpt_er[:self.frame])
            self.kalman_point_adam.set_data([self.k[self.frame]], [self.xOpt_adam[self.frame]])
            self.kalman_point_er.set_data([self.k[self.frame]], [self.xOpt_er[self.frame]])
        else:
            self.kalman_line.set_data(self.k[:self.frame], self.xOpt[:self.frame])
            self.kalman_point.set_data([self.k[self.frame]], [self.xOpt[self.frame]])

        self.canvas.draw_idle()
        self.frame += 1

    def run_experiment(self):
        N = self.N.value();
        a = self.a.value();
        T = self.T.value()
        motion_type = "simple" if self.model_classic.isChecked() else "physical"
        sigmaEta_vals, sigmaPsi_vals, sko_eta, sko_psi, error_map = kalman_full_experiment(N, a, T, motion_type)

        self.canvas.fig.clear()
        ax1 = self.canvas.fig.add_subplot(221)
        ax2 = self.canvas.fig.add_subplot(222)
        ax3 = self.canvas.fig.add_subplot(223)
        ax4 = self.canvas.fig.add_subplot(224)

        ax1.plot(sigmaEta_vals, sko_eta)
        ax1.axhline(1, color="red", linestyle="--", label="SKO = 1%")
        best_eta = sigmaEta_vals[np.argmin(sko_eta)]
        ax1.axvline(best_eta, linestyle="--")
        ax1.set_title(f"Error vs σ_eta (best={best_eta:.2f})")
        ax1.legend()

        ax2.plot(sigmaPsi_vals, sko_psi)
        ax2.axhline(1, color="red", linestyle="--", label="SKO = 1%")
        best_psi = sigmaPsi_vals[np.argmin(sko_psi)]
        ax2.axvline(best_psi, linestyle="--")
        ax2.set_title(f"Error vs σ_psi (best={best_psi:.2f})")
        ax2.legend()

        im = ax3.imshow(error_map, origin="lower", aspect="auto",
                        extent=[sigmaEta_vals.min(), sigmaEta_vals.max(), sigmaPsi_vals.min(), sigmaPsi_vals.max()])
        ax3.contour(sigmaEta_vals, sigmaPsi_vals, error_map, levels=[1], colors="red", linewidths=2)
        ax3.set_title("Error map")
        self.canvas.fig.colorbar(im, ax=ax3)

        ax4.axis("off")
        ax4.text(0.1, 0.6, "Valid region mapped", fontsize=14)
        self.canvas.draw()

    def run_changing_noise_experiment(self):
        N = self.N.value();
        a = self.a.value();
        T = self.T.value()
        motion_type = "simple" if self.model_classic.isChecked() else "physical"
        x, z, sigma = simulate_changing_noise_motion(N, a, T, motion_type)

        xOpt_k, DZ_k, SKO_k, _ = kalman_filter(x, z, N, a, T, 1, 1, motion_type)

        selected_model = self.model_adam if self.radio_adam.isChecked() else self.model_er
        xOpt_n, DZ_n, SKO_n, _ = neural_kalman_filter(x, z, N, a, T, selected_model, motion_type)

        t = np.arange(N)
        self.canvas.fig.clear()

        ax1 = self.canvas.fig.add_subplot(221)
        ax1.plot(t, x, color="black", label="True")
        ax1.plot(t, z, "--", color="magenta", label="Sensor")
        ax1.plot(t, xOpt_k, color="#00d4ff", label="Kalman")
        ax1.plot(t, xOpt_n, color="orange", label="Neural Kalman")
        ax1.legend()

        ax2 = self.canvas.fig.add_subplot(222)
        ax2.plot(t, sigma)
        ax2.set_title("Sensor noise σ(t)")

        ax3 = self.canvas.fig.add_subplot(223)
        ax3.plot(DZ_k, label="Kalman")
        ax3.plot(DZ_n, label="Neural")
        ax3.legend()

        ax4 = self.canvas.fig.add_subplot(224)
        ax4.axis("off")
        ax4.text(0.05, 0.2, f"Classical SKO = {SKO_k:.2f}\nNeural SKO = {SKO_n:.2f}", fontsize=12)
        self.canvas.draw()

    def training_error(self, err_msg):
        self.lbl_train_status.setText("Error! Check console.")
        self.btn_train.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KalmanApp()
    window.show()
    app.exec()