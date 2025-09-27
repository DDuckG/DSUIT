# src/utils/concurrency.py
import queue, threading

class StageQueue:
    def __init__(self, maxsize=8):
        self.q = queue.Queue(maxsize=maxsize)

    def put(self, item):
        self.q.put(item, block=True)

    def get(self):
        return self.q.get(block=True)

    def task_done(self):
        self.q.task_done()

class Worker(threading.Thread):
    def __init__(self, fn, name="worker"):
        super().__init__(daemon=True)
        self.fn = fn
        self.name = name

    def run(self):
        self.fn()
