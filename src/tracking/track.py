from kf import KalmanFilter
from iou import xyxy_to_xyah, xyah_to_xyxy

class Track:
    def __init__(self, bbox, score, cls, track_id, kf = None, max_age = 30):
        self.bbox = bbox
        self.score = score
        self.cls = cls
        self.track_id = track_id

        self.kf = kf if kf is not None else KalmanFilter()
        measurement = xyxy_to_xyah(bbox)
        self.mean, self.cov = self.kf.initiate(measurement)
        self.age = 1
        self.time_since_update = 0
        self.hits = 1
        self.max_age = max_age
        self.history = []

    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        pred_box = xyah_to_xyxy(self.mean[0:4])
        self.bbox = pred_box
        self.age += 1
        self.time_since_update += 1
        self.history.append(pred_box)
        return pred_box

    def update(self, bbox, score):
        meas = xyxy_to_xyah(bbox)
        self.mean, self.cov = self.kf.update(self.mean, self.cov, meas)
        self.bbox = bbox
        self.score = score
        self.time_since_update = 0
        self.hits += 1

    def mark_missed(self):
        pass

    def is_dead(self):
        return self.time_since_update > self.max_age