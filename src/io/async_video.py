# src/io/async_video.py
import cv2
import threading, queue, time
from dataclasses import dataclass

@dataclass
class FramePacket:
    frame_id: int
    rgb: any
    H: int
    W: int
    ts: float

class AsyncVideoReader:
    def __init__(self, path: str, prefetch: int = 8):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.q = queue.Queue(maxsize=int(prefetch))
        self._stop = False
        self.th = threading.Thread(target=self._worker, daemon=True)
        self.th.start()

    def _worker(self):
        fid = 0
        while not self._stop:
            ok, bgr = self.cap.read()
            if not ok:
                self.q.put(None)
                break
            fid += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pkt = FramePacket(frame_id=fid, rgb=rgb, H=self.H, W=self.W, ts=fid/self.fps)
            self.q.put(pkt)

    def __iter__(self):
        while True:
            pkt = self.q.get()
            if pkt is None: break
            yield pkt

    def release(self):
        self._stop = True
        self.th.join(timeout=0.2)
        self.cap.release()

class AsyncVideoWriter:
    def __init__(self, path: str, fps: float, size_wh, queue_size: int = 64):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, float(fps), (int(size_wh[0]), int(size_wh[1])))
        self.q = queue.Queue(maxsize=int(queue_size))
        self._stop = False
        self.th = threading.Thread(target=self._worker, daemon=True)
        self.th.start()

    def _worker(self):
        while not self._stop:
            item = self.q.get()
            if item is None: break
            self.writer.write(item)

    def write_rgb(self, rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.q.put(bgr)

    def release(self):
        self._stop = True
        self.q.put(None)
        self.th.join(timeout=0.2)
        self.writer.release()
