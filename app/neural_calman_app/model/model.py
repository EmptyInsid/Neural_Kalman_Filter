import numpy as np


def simulate_motion(N, a, T, sigmaPsi, sigmaEta):

    x = np.zeros(N)
    z = np.zeros(N)

    z[0] = x[0] + np.random.normal(0, sigmaEta)

    for t in range(N-1):
        x[t+1] = (
            x[t]
            + a*T*t
            + np.random.normal(0, sigmaPsi)
        )

        z[t+1] = (
            x[t+1]
            + np.random.normal(0, sigmaEta)
        )

    return x, z

def simulate_physical_motion(N, a, T, sigmaPsi, sigmaEta):

    x = np.zeros(N)
    v = np.zeros(N)
    z = np.zeros(N)

    v[0] = 0
    x[0] = 0

    z[0] = x[0] + np.random.normal(0, sigmaEta)

    for t in range(N-1):

        # истинное движение
        v[t+1] = v[t] + a*T + np.random.normal(0, sigmaPsi)

        x[t+1] = (
            x[t]
            + v[t]*T
            + 0.5*a*T**2
        )

        # измерение сенсора
        z[t+1] = x[t+1] + np.random.normal(0, sigmaEta)

    return x, z

def kalman_filter(
        x,
        z,
        N,
        a,
        T,
        sigmaPsi,
        sigmaEta,
        motion_type
):

    xOpt = np.zeros(N)
    eOpt = np.zeros(N)
    K = np.zeros(N)

    vOpt = np.zeros(N)

    xOpt[0] = z[0]
    eOpt[0] = sigmaEta

    for t in range(N-1):

        # prediction step
        if motion_type == "simple":

            x_pred = xOpt[t] + a*T*t

        elif motion_type == "physical":

            vOpt[t+1] = vOpt[t] + a*T

            x_pred = (
                xOpt[t]
                + vOpt[t]*T
                + 0.5*a*T**2
            )

        # Kalman gain
        eOpt[t+1] = np.sqrt(
            (sigmaEta**2 * (eOpt[t]**2 + sigmaPsi**2))
            /
            (sigmaEta**2 + eOpt[t]**2 + sigmaPsi**2)
        )

        K[t+1] = (eOpt[t+1]**2) / sigmaEta**2

        # update
        xOpt[t+1] = (
            x_pred*(1-K[t+1])
            + K[t+1]*z[t+1]
        )

    err = xOpt[:-1] - x[:-1]

    DZ = err * 100 / (np.max(x) - np.min(x))

    SKO = np.std(DZ)

    return xOpt, DZ, SKO, K


def kalman_full_experiment(N, a, T, motion_type):

    sigmaPsi_const = 1.0
    sigmaEta_const = 1.0

    sigmaEta_vals = np.linspace(0.1, 100, 30)
    sigmaPsi_vals = np.linspace(0.1, 100, 30)

    sko_eta = []
    sko_psi = []

    # sigmaPsi фиксирован, меняем sigmaEta
    for sigmaEta in sigmaEta_vals:

        if motion_type == "physical":
            x, z = simulate_physical_motion(N, a, T, sigmaPsi_const, sigmaEta)
        else:
            x, z = simulate_motion(N, a, T, sigmaPsi_const, sigmaEta)

        xOpt, DZ, SKO, K = kalman_filter(
            x, z, N, a, T, sigmaPsi_const, sigmaEta, motion_type
        )

        sko_eta.append(SKO)

    # sigmaEta фиксирован, меняем sigmaPsi
    for sigmaPsi in sigmaPsi_vals:

        if motion_type == "physical":
            x, z = simulate_physical_motion(N, a, T, sigmaPsi_const, sigmaEta)
        else:
            x, z = simulate_motion(N, a, T, sigmaPsi_const, sigmaEta)

        xOpt, DZ, SKO, K = kalman_filter(
            x, z, N, a, T, sigmaPsi, sigmaEta_const, motion_type
        )

        sko_psi.append(SKO)

    # карта ошибок
    error_map = np.zeros((len(sigmaPsi_vals), len(sigmaEta_vals)))

    for i, sigmaPsi in enumerate(sigmaPsi_vals):
        for j, sigmaEta in enumerate(sigmaEta_vals):

            if motion_type == "physical":
                x, z = simulate_physical_motion(N, a, T, sigmaPsi_const, sigmaEta)
            else:
                x, z = simulate_motion(N, a, T, sigmaPsi_const, sigmaEta)

            xOpt, DZ, SKO, K = kalman_filter(
                x, z, N, a, T, sigmaPsi, sigmaEta, motion_type
            )

            error_map[i, j] = SKO

    return (
        sigmaEta_vals,
        sigmaPsi_vals,
        np.array(sko_eta),
        np.array(sko_psi),
        error_map
    )


def simulate_changing_noise_motion(
        N,
        a,
        T,
        motion_type
):

    x = np.zeros(N)
    z = np.zeros(N)
    sigma = np.zeros(N)

    v = np.zeros(N)

    sigmaPsi = 1

    for t in range(N-1):

        # changing sensor noise
        if t < N//3:
            sigmaEta = 1

        elif t < 2*N//3:
            sigmaEta = 5

        else:
            sigmaEta = 12

        sigma[t] = sigmaEta

        # motion model
        if motion_type == "simple":

            x[t+1] = (
                x[t]
                + a*T*t
                + np.random.normal(0, sigmaPsi)
            )

        elif motion_type == "physical":

            v[t+1] = v[t] + a*T + np.random.normal(0, sigmaPsi)

            x[t+1] = (
                x[t]
                + v[t]*T
                + 0.5*a*T**2
            )

        # sensor
        z[t+1] = (
            x[t+1]
            + np.random.normal(0, sigmaEta)
        )

    return x, z, sigma
