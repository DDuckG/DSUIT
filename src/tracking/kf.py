import numpy as np

class KalmanFilter:
    def __init__(self):
        self._dim_x = 7
        self._dim_z = 4

        self.F = np.eye(self._dim_x)
        dt = 1.0
        for i in range(3):
            self.F[i, i+4] = dt

        self.H = np.zeros((self._dim_z, self._dim_x))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.P = np.eye(self._dim_x) * 10.0
        self.R = np.eye(self._dim_z) * 1.0
        self.Q = np.eye(self._dim_x) * 0.01

    def initiate(self, measurement):
        x = np.zeros((self._dim_x,))
        x[0:4] = measurement
        x[4:] = 0.0
        return x, self.P.copy()

    def predict(self, x, P):
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(self, x, P, z):
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        y = z - (self.H @ x)
        x_upd = x + K @ y
        P_upd = (np.eye(self._dim_x) - K @ self.H) @ P
        return x_upd, P_upd

    def gating_distance(self, x, P, measurements):
        mean = self.H @ x
        S = self.H @ P @ self.H.T + self.R
        d = measurements - mean[np.newaxis, :]
        invS = np.linalg.inv(S)
        m = np.sum(d @ invS * d, axis=1)
        return m