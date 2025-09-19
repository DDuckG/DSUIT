import numpy as np
from track import Track
from iou import iou_batch
from linear_assignment import linear_assignment

class OCSort:
    def __init__(self, max_age = 30, iou_threshold = 0.3, reid_iou = 0.5):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.reid_iou = reid_iou
        self.tracks = []
        self._next_id = 1

    def _predict(self):
        for track in self.tracks:
            track.predict()

    def update(self, detections, scores, classes):
        # 1. predict all tracks
        self._predict()

        # 2. build IoU cost matrix
        if len(self.tracks) == 0:
            unmatched_dets = list(range(len(detections)))
            matches = []
            unmatched_tracks = []
        else:
            track_bboxes = np.array([track.bbox for track in self.tracks])
            det_bboxes = np.array(detections)
            iou_mat = iou_batch(track_bboxes, det_bboxes)

            for ti, tr in enumerate(self.tracks):
                for di, cls in enumerate(classes):
                    if tr.cls != cls:
                        iou_mat[ti,di] = 0.0

            cost_matrix = -iou_mat
            large_cost = 1000000
            cost_matrix[iou_mat < self.iou_threshold] = large_cost

            row_ind, col_ind = linear_assignment(cost_matrix)
            matches = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets = list(range(len(detections)))
            for row, col in zip(row_ind, col_ind):
                if cost_matrix[row,col] >= large_cost:
                    continue
                matches.append((row,col))
                unmatched_tracks.remove(row)
                unmatched_dets.remove(col)

        # 3. update matched tracks
        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            track.update(detections[d_idx], scores[d_idx])

        # 4. create new tracks for unmatched detections
        for di in unmatched_dets:
            new_track = Track(detections[di], scores[di], classes[di], self._next_id, max_age = self.max_age)
            self._next_id += 1
            self.tracks.append(new_track)

        # 5. mark unmatched tracks missed and remove dead ones
        for ti in unmatched_tracks:
            tr = self.tracks[ti]
            tr.time_since_update += 1
        self.tracks = [track for track in self.tracks if not track.is_dead()]

        # 6. return current active tracks
        active = []
        for track in self.tracks:
            if track.time_since_update == 0:
                active.append((track.track_id, track.bbox, track.score, track.cls))
        return active
