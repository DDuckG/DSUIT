# src/utils/streams.py
import torch

class CUDAStreams:
    def __init__(self, n=2):
        self.enabled = torch.cuda.is_available()
        self.streams = [torch.cuda.Stream() for _ in range(n)] if self.enabled else []

    def get(self, idx=0):
        if self.enabled: return self.streams[idx % len(self.streams)]
        return None

    def use(self, idx=0):
        if self.enabled:
            return torch.cuda.stream(self.streams[idx % len(self.streams)])
        # CPU fallback
        from contextlib import nullcontext
        return nullcontext()
