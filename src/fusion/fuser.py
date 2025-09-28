# src/fusion/fuser.py
import numpy as np
import torch

# ----------------------- utils -----------------------

def _is_torch(x):
    return isinstance(x, torch.Tensor)

def _to_torch(x, dtype=torch.float32, device=None):
    if x is None:
        return None
    if _is_torch(x):
        return x.to(device=device, dtype=dtype) if device is not None else x.to(dtype=dtype)
    # ưu tiên GPU nếu có
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.as_tensor(x, dtype=dtype, device=dev)

def _to_torch_i64(x, device=None):
    if x is None:
        return None
    if _is_torch(x):
        return x.to(device=device, dtype=torch.long) if device is not None else x.to(dtype=torch.long)
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.as_tensor(x, dtype=torch.long, device=dev)

def _to_numpy(x, dtype=None):
    if _is_torch(x):
        a = x.detach().cpu().numpy()
        return a.astype(dtype) if dtype is not None else a
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a

def _empty_numpy(shape, dtype=float):
    return np.zeros(shape, dtype=dtype)

def _area(xyxy_t):
    # xyxy_t: (...,4)
    w = (xyxy_t[..., 2] - xyxy_t[..., 0]).clamp(min=0.0)
    h = (xyxy_t[..., 3] - xyxy_t[..., 1]).clamp(min=0.0)
    return w * h

# ----------------------- IoU / NMS -----------------------

@torch.no_grad()
def iou_with(a, arr):
    """
    a: (4,) torch/np ; arr: (N,4) torch/np
    return: (N,) IoU (torch tensor on same device if torch inputs else numpy)
    """
    # chuẩn hoá về torch (ưu tiên device của arr nếu là tensor)
    dev = arr.device if _is_torch(arr) else (a.device if _is_torch(a) else ("cuda" if torch.cuda.is_available() else "cpu"))
    at = _to_torch(a, dtype=torch.float32, device=dev).view(1, 4)
    bt = _to_torch(arr, dtype=torch.float32, device=dev).view(-1, 4)

    x1 = torch.maximum(at[:, 0], bt[:, 0])
    y1 = torch.maximum(at[:, 1], bt[:, 1])
    x2 = torch.minimum(at[:, 2], bt[:, 2])
    y2 = torch.minimum(at[:, 3], bt[:, 3])

    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    ua = _area(at).view(-1) + _area(bt) - inter + 1e-9
    iou = inter / ua
    # trả cùng kiểu bên ngoài (numpy để giữ tương thích cũ)
    return _to_numpy(iou, dtype=np.float32)

@torch.no_grad()
def _iou_matrix(A_t, B_t):
    """
    A_t: (N,4) torch, B_t: (M,4) torch  -> (N,M) torch
    """
    if A_t.numel() == 0 or B_t.numel() == 0:
        return torch.zeros((A_t.shape[0], B_t.shape[0]), device=A_t.device, dtype=torch.float32)
    x1 = torch.maximum(A_t[:, None, 0], B_t[None, :, 0])
    y1 = torch.maximum(A_t[:, None, 1], B_t[None, :, 1])
    x2 = torch.minimum(A_t[:, None, 2], B_t[None, :, 2])
    y2 = torch.minimum(A_t[:, None, 3], B_t[None, :, 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    ua = _area(A_t)[:, None] + _area(B_t)[None, :] - inter + 1e-9
    return inter / ua

@torch.no_grad()
def nms_iou(xyxy, scores, iou_thr=0.55):
    """
    xyxy, scores: torch/np. Trả về list chỉ số keep (python int).
    """
    if xyxy is None or len(xyxy) == 0:
        return []

    # về torch (GPU nếu có)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    boxes_t = _to_torch(xyxy, dtype=torch.float32, device=dev).view(-1, 4)
    scores_t = _to_torch(scores, dtype=torch.float32, device=dev).view(-1)
    order = torch.argsort(scores_t, descending=True)

    keep = []
    while int(order.numel()) > 0:
        i = int(order[0].item())
        keep.append(i)
        if int(order.numel()) == 1:
            break
        rest = order[1:]
        ious = _iou_matrix(boxes_t[i:i+1, :], boxes_t[rest, :]).squeeze(0)  # (K,)
        mask = ious <= float(iou_thr)
        order = rest[mask]

    return keep

# ----------------------- Merge YOLO & OBS -----------------------

@torch.no_grad()
def merge_yolo_obstacle(yolo_boxes, yolo_scores, yolo_cls, yolo_dists,
                        obs_boxes, obs_scores, obs_dists, iou_merge=0.5):
    """
    Gộp kết quả YOLO (+dist đo từ depth) với các obstacle chung chung (edge/height).
    - Nếu IoU(obs, yolo_max) >= iou_merge -> dùng bbox YOLO, class=YOLO, flag=True.
    - Ngược lại giữ bbox OBS, class=-1, flag=False.
    - Các YOLO chưa dùng sẽ được thêm vào sau.
    Tất cả tính toán trên GPU nếu có; trả về numpy để tiện vẽ.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Về torch
    yb = _to_torch(yolo_boxes, dtype=torch.float32, device=dev).view(-1, 4) if yolo_boxes is not None and len(yolo_boxes) else torch.zeros((0,4), device=dev)
    ys = _to_torch(yolo_scores, dtype=torch.float32, device=dev).view(-1)    if yolo_scores is not None and len(yolo_scores) else torch.zeros((0,), device=dev)
    yc = _to_torch_i64(yolo_cls, device=dev).view(-1)                        if yolo_cls    is not None and len(yolo_cls)    else torch.zeros((0,), dtype=torch.long, device=dev)
    yd = _to_torch(yolo_dists, dtype=torch.float32, device=dev).view(-1)     if yolo_dists  is not None and len(yolo_dists)  else torch.zeros((0,), device=dev)

    ob = _to_torch(obs_boxes, dtype=torch.float32, device=dev).view(-1, 4)   if obs_boxes   is not None and len(obs_boxes)   else torch.zeros((0,4), device=dev)
    os = _to_torch(obs_scores, dtype=torch.float32, device=dev).view(-1)     if obs_scores  is not None and len(obs_scores)  else torch.zeros((0,), device=dev)
    od = _to_torch(obs_dists, dtype=torch.float32, device=dev).view(-1)      if obs_dists   is not None and len(obs_dists)   else torch.zeros((0,), device=dev)

    boxes = []
    scores = []
    clss = []
    flags = []
    dists = []

    yused = torch.zeros((yb.shape[0],), dtype=torch.bool, device=dev)

    # duyệt OBS -> match YOLO gần nhất theo IoU
    for i in range(ob.shape[0]):
        b = ob[i:i+1, :]  # (1,4)
        if yb.shape[0] == 0:
            boxes.append(ob[i])
            scores.append(os[i] if os.numel() else torch.tensor(0.0, device=dev))
            clss.append(torch.tensor(-1, device=dev))
            flags.append(torch.tensor(False, device=dev))
            dists.append(od[i] if od.numel() else torch.tensor(float('nan'), device=dev))
            continue

        ious = _iou_matrix(b, yb).squeeze(0)  # (Ny,)
        j = int(torch.argmax(ious).item())
        if float(ious[j].item()) >= float(iou_merge):
            yused[j] = True
            # lấy box, score, cls từ YOLO; distance: ưu tiên YOLO nếu finite
            dist = yd[j] if (yd.numel() and torch.isfinite(yd[j])) else (od[i] if od.numel() else torch.tensor(float('nan'), device=dev))
            sc = torch.maximum(ys[j] if ys.numel() else torch.tensor(0.0, device=dev),
                               os[i] if os.numel() else torch.tensor(0.0, device=dev))
            boxes.append(yb[j])
            scores.append(sc)
            clss.append(yc[j] if yc.numel() else torch.tensor(-1, device=dev))
            flags.append(torch.tensor(True, device=dev))
            dists.append(dist)
        else:
            boxes.append(ob[i])
            scores.append(os[i] if os.numel() else torch.tensor(0.0, device=dev))
            clss.append(torch.tensor(-1, device=dev))
            flags.append(torch.tensor(False, device=dev))
            dists.append(od[i] if od.numel() else torch.tensor(float('nan'), device=dev))

    # thêm các YOLO chưa dùng
    for j in range(yb.shape[0]):
        if not bool(yused[j].item()):
            boxes.append(yb[j])
            scores.append(ys[j] if ys.numel() else torch.tensor(0.0, device=dev))
            clss.append(yc[j] if yc.numel() else torch.tensor(-1, device=dev))
            flags.append(torch.tensor(True, device=dev))
            dists.append(yd[j] if yd.numel() else torch.tensor(float('nan'), device=dev))

    if len(boxes) == 0:
        return (_empty_numpy((0, 4), dtype=np.float32),
                _empty_numpy((0,), dtype=np.float32),
                _empty_numpy((0,), dtype=np.int32),
                _empty_numpy((0,), dtype=bool),
                _empty_numpy((0,), dtype=np.float32))

    # stack -> numpy
    boxes_t = torch.stack(boxes, dim=0).to(torch.float32)
    scores_t = torch.stack(scores, dim=0).to(torch.float32)
    clss_t = torch.stack(clss, dim=0).to(torch.long)
    flags_t = torch.stack(flags, dim=0).to(torch.bool)
    dists_t = torch.stack(dists, dim=0).to(torch.float32)

    return (_to_numpy(boxes_t, dtype=np.float32),
            _to_numpy(scores_t, dtype=np.float32),
            _to_numpy(clss_t,  dtype=np.int32),
            _to_numpy(flags_t, dtype=bool),
            _to_numpy(dists_t, dtype=np.float32))
