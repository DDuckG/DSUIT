# src/io/video_writer.py
import cv2

class VideoWriter:
    def __init__(self, path: str, fps: float, size_wh):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, float(fps), (int(size_wh[0]), int(size_wh[1])))

    def write_rgb(self, rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)

    def release(self):
        self.writer.release()
