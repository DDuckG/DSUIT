# src/io/video_reader.py
import cv2
from dataclasses import dataclass

@dataclass
class FramePacket:
    frame_id: int
    rgb: any
    H: int
    W: int
    ts: float

class VideoReader:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self):
        fid = 0
        while True:
            ok, bgr = self.cap.read()
            if not ok: break
            fid += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yield FramePacket(frame_id=fid, rgb=rgb, H=self.H, W=self.W, ts=fid/self.fps)

    def release(self):
        self.cap.release()
