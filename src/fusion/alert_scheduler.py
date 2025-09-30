import json
import math
import os
from typing import List, Tuple, Optional
import numpy as np


def _iou_xyxy(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return (inter / (ua + ub - inter + 1e-9)) if (ua > 0 and ub > 0) else 0.0


class AlertScheduler:
    def __init__(
        self,
        video_fps: float,
        image_wh: Tuple[int, int],
        roi_frac: Tuple[float, float, float, float] = (0.25, 0.25, 0.5, 0.5),
        roi_iou_min: float = 0.10,
        thr_low_m: float = 1.8,
        thr_high_m: float = 3.5,
        beep_interval_s: float = 2.0,
        debounce_frames: int = 2,
        hysteresis_frames: int = 2,
        log_texts: Optional[dict] = None,
    ):
        self.fps = float(video_fps)
        self.W, self.H = int(image_wh[0]), int(image_wh[1])
        rx, ry, rw, rh = roi_frac
        self.roi = (
            rx * self.W,
            ry * self.H,
            (rx + rw) * self.W,
            (ry + rh) * self.H
        )
        self.roi_iou_min = float(roi_iou_min)
        self.tlow = float(thr_low_m)
        self.thigh = float(thr_high_m)
        self.beep_interval = float(beep_interval_s)
        self.db_frames = int(debounce_frames)
        self.hys_frames = int(hysteresis_frames)
        self.texts = {"warn": "Be careful", "danger": "Danger"}
        if isinstance(log_texts, dict):
            self.texts.update({k: str(v) for k, v in log_texts.items() if k in ("warn","danger")})

        # FSM
        self.state = "none"   # "none" | "warn" | "danger"
        self.cand_state = "none"
        self.cand_count = 0
        self.seg_start_t = None
        self.next_beep_at = math.inf

        self.events: List[dict] = []     # {t, level, message, bbox?, dist?}
        self.segments: List[dict] = []   # {start, end, state}

        self.meta = {
            "video_fps": self.fps,
            "frame_size": [self.W, self.H],
            "roi_frac": [float(x) for x in (roi_frac)],
            "roi_px": [float(x) for x in self.roi],
            "thr_m": {"low": self.tlow, "high": self.thigh},
            "beep_interval_s": self.beep_interval,
        }

    def _inside_roi(self, box: np.ndarray) -> bool:
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        if (self.roi[0] <= cx <= self.roi[2]) and (self.roi[1] <= cy <= self.roi[3]):
            return True
        return _iou_xyxy((x1, y1, x2, y2), self.roi) >= self.roi_iou_min

    def _level_from_dist(self, d: float) -> str:
        if not np.isfinite(d) or d <= 0:
            return "none"
        if d < self.tlow:
            return "danger"
        if d <= self.thigh:
            return "warn"
        return "none"

    def _emit_event(self, t: float, level: str, bbox: Optional[np.ndarray], dist: float, source: str = ""):
        ev = {
            "t": float(max(0.0, t)),
            "level": level,
            "message": self.texts[level],
            "dist": float(dist) if np.isfinite(dist) else None,
            "bbox": [float(v) for v in bbox] if bbox is not None else None,
            "source": source or "",
        }
        self.events.append(ev)

    def _open_segment_if_needed(self, t: float):
        if self.seg_start_t is None and self.state in ("warn", "danger"):
            self.seg_start_t = float(t)

    def _close_segment_if_needed(self, t: float):
        if self.seg_start_t is not None and self.state in ("warn", "danger"):
            self.segments.append({"start": self.seg_start_t, "end": float(t), "state": self.state})
            self.seg_start_t = None

    def feed(self, out_frame_id: int, boxes: np.ndarray, dists: np.ndarray,
             now_src: str = "", candidates_src: Optional[List[str]] = None):
        t = float(out_frame_id / max(1e-6, self.fps))
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        dists = np.asarray(dists, dtype=np.float32).reshape(-1)
        N = min(len(boxes), len(dists))
        if candidates_src is None:
            sources = [""] * N
        else:
            sources = [str(s) for s in candidates_src[:N]]

        best = {"warn": None, "danger": None}  # (idx, dist)
        for i in range(N):
            d = float(dists[i])
            level = self._level_from_dist(d)
            if level == "none":
                continue
            if not self._inside_roi(boxes[i]):
                continue
            if best[level] is None or d < best[level][1]:
                best[level] = (i, d)

        present_d = best["danger"] is not None
        present_w = best["warn"] is not None

        desired = "none"
        chosen_idx = None
        if present_d:
            desired = "danger"; chosen_idx = best["danger"][0]
        elif present_w:
            desired = "warn";   chosen_idx = best["warn"][0]

        if desired != self.state:
            if desired == self.cand_state:
                self.cand_count += 1
            else:
                self.cand_state = desired
                self.cand_count = 1

            need_frames = self.db_frames
            # hạ cấp từ danger -> warn cứng tay hơn một chút
            if self.state == "danger" and desired == "warn":
                need_frames = max(self.db_frames, self.hys_frames)

            if self.cand_count >= need_frames:
                # chuyển trạng thái chính thức
                # đóng segment cũ (nếu có)
                self._close_segment_if_needed(t)
                prev = self.state
                self.state = desired
                self.cand_state = "none"
                self.cand_count = 0
                # mở segment mới (nếu có)
                self._open_segment_if_needed(t)

                if self.state in ("warn", "danger"):
                    idx = chosen_idx if chosen_idx is not None else (best[self.state][0] if best[self.state] else None)
                    b = boxes[idx] if (idx is not None) else None
                    d = dists[idx] if (idx is not None) else float("nan")
                    self._emit_event(t, self.state, b, d, now_src)
                    self.next_beep_at = t + self.beep_interval
                else:
                    self.next_beep_at = math.inf

        else:
            if self.state in ("warn", "danger") and t + 1e-3 >= self.next_beep_at:
                idx = chosen_idx if chosen_idx is not None else (best[self.state][0] if best[self.state] else None)
                b = boxes[idx] if (idx is not None) else None
                d = dists[idx] if (idx is not None) else float("nan")
                self._emit_event(t, self.state, b, d, now_src)
                self.next_beep_at += self.beep_interval

    def finalize(self, total_frames: int, out_json: str):
        T = float(total_frames / max(1e-6, self.fps))
        self._close_segment_if_needed(T)

        data = {
            "meta": self.meta,
            "segments": self.segments,
            "events": sorted(self.events, key=lambda e: e["t"]),
        }
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
