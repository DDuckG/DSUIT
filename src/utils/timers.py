# src/utils/timers.py
import time

class FPSMeter:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.t0 = time.time()
        self.fps = 0.0
        self.count = 0

    def tick(self):
        t = time.time()
        dt = t - self.t0
        self.t0 = t
        self.count += 1
        if dt > 0:
            f = 1.0/dt
            self.fps = self.alpha*self.fps + (1-self.alpha)*f
        return self.fps
