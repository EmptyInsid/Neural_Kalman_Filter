import sys
import numpy as np

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QRadioButton,
    QPushButton, QDoubleSpinBox, QSpinBox, QFormLayout, QGroupBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
# plt.style.use("dark_background")

from app.classic_calman_app.model.model import (
    simulate_motion,
    simulate_physical_motion,
    kalman_filter,
    kalman_full_experiment
)


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self):

        self.fig = Figure(figsize=(16,8))

        super().__init__(self.fig)


class KalmanApp(QWidget):

    def __init__(self):

        super().__init__()

        # ===== настройка окна приложения =====
        self.resize(1800, 1200)
        self.setWindowTitle("Kalman Filter Demo")
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #dddddd;
                font-size: 12pt;
            }

            QGroupBox {
                border: 1px solid #444;
                margin-top: 10px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }

            QPushButton {
                background-color: #2d89ef;
                color: white;
                border-radius: 6px;
                padding: 8px;
            }

            QPushButton:hover {
                background-color: #4aa3ff;
            }

            QPushButton:pressed {
                background-color: #1b5fa7;
            }

            QSpinBox, QDoubleSpinBox {
                background-color: #2a2a2a;
                border: 1px solid #444;
                padding: 3px;
            }
        """)

        # ===== настройка анимации =====
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)

        main_layout = QHBoxLayout()
        control_panel = QVBoxLayout()

        # ===== параметры =====

        form = QFormLayout()

        self.N = QSpinBox()
        self.N.setRange(10, 10000)
        self.N.setValue(100)

        self.a = QDoubleSpinBox()
        self.a.setRange(-10, 10)
        self.a.setDecimals(3)
        self.a.setValue(0.15)

        self.T = QDoubleSpinBox()
        self.T.setRange(-10, 10)
        self.T.setDecimals(3)
        self.T.setValue(-0.5)

        self.sigmaPsi = QDoubleSpinBox()
        self.sigmaPsi.setRange(0, 100)
        self.sigmaPsi.setValue(1)

        self.sigmaEta = QDoubleSpinBox()
        self.sigmaEta.setRange(0, 100)
        self.sigmaEta.setValue(4)

        form.addRow("Number of steps (N)", self.N)
        form.addRow("Acceleration (a)", self.a)
        form.addRow("Time step (T)", self.T)
        form.addRow("Model noise σψ", self.sigmaPsi)
        form.addRow("Sensor noise ση", self.sigmaEta)

        params_box = QGroupBox("Simulation parameters")
        params_box.setLayout(form)

        control_panel.addWidget(params_box)

        # кнопки

        self.model_classic = QRadioButton("Simple model")
        self.model_physics = QRadioButton("Physical motion")

        self.model_classic.setChecked(True)

        control_panel.addWidget(self.model_classic)
        control_panel.addWidget(self.model_physics)


        self.button = QPushButton("Run simulation")
        self.button.clicked.connect(self.run_simulation)

        control_panel.addWidget(self.button)
        control_panel.addStretch()

        self.exp_button = QPushButton("Run parameter experiment")
        self.exp_button.clicked.connect(self.run_experiment)

        control_panel.addWidget(self.exp_button)

        # график

        self.canvas = MplCanvas()
        main_layout.addLayout(control_panel, 1)
        main_layout.addWidget(self.canvas, 4)

        self.setLayout(main_layout)


    def run_simulation(self):

        N = self.N.value()
        a = self.a.value()
        T = self.T.value()
        sigmaPsi = self.sigmaPsi.value()
        sigmaEta = self.sigmaEta.value()

        if self.model_classic.isChecked():

            self.x, self.z = simulate_motion(
                N, a, T, sigmaPsi, sigmaEta
            )

        else:

            self.x, self.z = simulate_physical_motion(
                N, a, T, sigmaPsi, sigmaEta
            )


        self.xOpt, self.DZ, self.SKO, self.K = kalman_filter(
            self.x, self.z, N, a, T, sigmaPsi, sigmaEta
        )

        self.k = np.arange(N)

        self.frame = 0

        self.canvas.fig.clear()

        self.canvas.fig.clear()

        self.canvas.ax1 = self.canvas.fig.add_subplot(211)
        self.canvas.ax2 = self.canvas.fig.add_subplot(212)

        # self.ax3 = self.fig.add_subplot(313)

        self.timer.start(16)

        # график траектории движения
        ax = self.canvas.ax1
        ax.set_title("Object Tracking")
        ax.set_ylabel("position")

        ax.set_xlim(0, len(self.k))
        ax.set_ylim(
            min(self.x.min(), self.z.min()) - 5,
            max(self.x.max(), self.z.max()) + 5
        )

        model_name = (
            "Simple motion model"
            if self.model_classic.isChecked()
            else "Physical motion model"
        )

        ax.set_title(f"Object Tracking ({model_name})")


        # линии истории
        self.true_line, = ax.plot([], [], color="black", label="True trajectory")
        self.sensor_line, = ax.plot([], [], "--", color="#ea02ff", label="Sensor")
        self.kalman_line, = ax.plot([], [], color="#00d4ff", label="Kalman")

        # маркеры текущего положения
        self.true_point, = ax.plot([], [], "o", color="black", markersize=8)
        self.sensor_point, = ax.plot([], [], "x", color="#ea02ff", markersize=8)
        self.kalman_point, = ax.plot([], [], "o", color="#00d4ff", markersize=8)

        ax.legend()

        # график ошибки

        self.canvas.ax2.plot(self.DZ)
        self.canvas.ax2.set_title(f"Error %, SKO={self.SKO:.2f}")

        # график коэффициента калмана
        
        # self.canvas.ax3.plot(self.K)
        # self.canvas.ax3.set_title("Kalman Gain")
        # self.canvas.ax3.set_ylabel("K")
        # self.canvas.ax3.set_xlabel("time")


        self.canvas.draw()
    
    def update_animation(self):
        if self.frame >= len(self.k):
            self.timer.stop()
            return

        # обновляем линии истории
        self.true_line.set_data(
            self.k[:self.frame],
            self.x[:self.frame]
        )

        self.sensor_line.set_data(
            self.k[:self.frame],
            self.z[:self.frame]
        )

        self.kalman_line.set_data(
            self.k[:self.frame],
            self.xOpt[:self.frame]
        )

        # текущие точки
        self.true_point.set_data(
            [self.k[self.frame]],
            [self.x[self.frame]]
        )

        self.sensor_point.set_data(
            [self.k[self.frame]],
            [self.z[self.frame]]
        )

        self.kalman_point.set_data(
            [self.k[self.frame]],
            [self.xOpt[self.frame]]
        )

        self.canvas.draw_idle()

        self.frame += 1

    def run_experiment(self):
        N = self.N.value()
        a = self.a.value()
        T = self.T.value()

        model_type = (
            "simple"
            if self.model_classic.isChecked()
            else "physical"
        )

        (
            sigmaEta_vals,
            sigmaPsi_vals,
            sko_eta,
            sko_psi,
            error_map
        ) = kalman_full_experiment(N, a, T, model_type)

        self.canvas.fig.clear()

        ax1 = self.canvas.fig.add_subplot(221)
        ax2 = self.canvas.fig.add_subplot(222)
        ax3 = self.canvas.fig.add_subplot(223)
        ax4 = self.canvas.fig.add_subplot(224)

        # график sigmaEta
        ax1.plot(sigmaEta_vals, sko_eta)
        ax1.axhline(1, color="red", linestyle="--", label="SKO = 1%")
        ax1.legend()


        best_idx = np.argmin(sko_eta)
        best_eta = sigmaEta_vals[best_idx]

        ax1.axvline(best_eta, linestyle="--")

        ax1.set_title(f"Error vs σ_eta (best={best_eta:.2f}) (σ_psi=1.0)")
        ax1.set_xlabel("sigmaEta")
        ax1.set_ylabel("SKO")

        # график sigmaPsi
        ax2.plot(sigmaPsi_vals, sko_psi)
        ax2.axhline(1, color="red", linestyle="--", label="SKO = 1%")
        ax2.legend()

        best_idx = np.argmin(sko_psi)
        best_psi = sigmaPsi_vals[best_idx]

        ax2.axvline(best_psi, linestyle="--")

        ax2.set_title(f"Error vs σ_psi (best={best_psi:.2f}), (σ_eta=1.0)")
        ax2.set_xlabel("sigmaPsi")
        ax2.set_ylabel("SKO")

        # карта ошибок
        im = ax3.imshow(
            error_map,
            origin="lower",
            aspect="auto",
            extent=[
                sigmaEta_vals.min(),
                sigmaEta_vals.max(),
                sigmaPsi_vals.min(),
                sigmaPsi_vals.max()
            ]
        )

        ax3.contour(
            sigmaEta_vals,
            sigmaPsi_vals,
            error_map,
            levels=[1],
            colors="red",
            linewidths=2
        )


        ax3.set_title("Error map")
        ax3.set_xlabel("sigmaEta")
        ax3.set_ylabel("sigmaPsi")

        self.canvas.fig.colorbar(im, ax=ax3)

        # 4️⃣ таблица лучших значений
        # table_data = []

        # for i in range(15):
        #     idx = np.unravel_index(
        #         np.argmin(error_map),
        #         error_map.shape
        #     )

        #     psi = sigmaPsi_vals[idx[0]]
        #     eta = sigmaEta_vals[idx[1]]
        #     sko = error_map[idx]

        #     table_data.append([
        #         f"{psi:.4f}",
        #         f"{eta:.4f}",
        #         f"{sko:.4f}"
        #     ])


        #     error_map[idx] = np.inf

        # ax4.axis("off")

        # table = ax4.table(
        #     cellText=table_data,
        #     colLabels=["sigmaPsi", "sigmaEta", "SKO"],
        #     loc="center"
        # )

        # print(table_data)

        # ax4.set_title("Best parameter sets")

        valid_pairs = []

        for i in range(len(sigmaPsi_vals)):
            for j in range(len(sigmaEta_vals)):

                if error_map[i, j] <= 1:

                    valid_pairs.append(
                        (sigmaPsi_vals[i], sigmaEta_vals[j], error_map[i, j])
                    )
        psi_vals = [v[0] for v in valid_pairs]
        eta_vals = [v[1] for v in valid_pairs]

        psi_min = min(psi_vals)
        psi_max = max(psi_vals)

        eta_min = min(eta_vals)
        eta_max = max(eta_vals)

        ax4.axis("off")

        text = f"""
        Valid region (SKO ≤ 1%)

        σψ ∈ [{psi_min:.2f} , {psi_max:.2f}]
        ση ∈ [{eta_min:.2f} , {eta_max:.2f}]

        Valid parameter sets: {len(valid_pairs)}
        """

        ax4.text(
            0.1,
            0.6,
            text,
            fontsize=14
        )

        self.canvas.draw()



app = QApplication(sys.argv)

window = KalmanApp()
window.show()

app.exec()
